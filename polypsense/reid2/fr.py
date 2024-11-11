# %%
import argparse
import json
import os
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

from polypsense.dataset.realcolon import (
    get_frame_id,
    get_info,
    load_files_list,
    parsevocfile,
)
from polypsense.mve.trackleter import Trackleter
from polypsense.reid2.backbone import get_backbone
from polypsense.reid2.clustering import associate, get_clustering
from polypsense.reid2.dataset import FragmentationRateDataset, get_polyps
from polypsense.reid2.retrieval import (
    get_labels,
)  # TODO: move `get_labels`, `get_features`


def run(
    exp_name: str,
    data_path: str,
    output_path: str,
    encoder_type: str,
    encoder_ckpt: str,
    encoder_kwargs: dict,
    clustering_type: str,
    clustering_hparams: dict | None,
    tracklet_aggregation_method: str | None,
    split: str | None,
    video_names: list[str] | None,
    target_fpr: float = 0.05,
    # By default we used to compute tp, fp, fn, tn for each video and then sum
    # them up to compute metrics such as false positive rate. This, seems to
    # fall into the category of "micro" averaging. Hoowever, this case may be
    # dominated by videos with many tracklets. (We found, for example, the video
    # 003-013 to have ore than 400 tracklet and anextremely high number of false
    # positives). On the other hand, "macro" averaging actually compute metrics
    # per video and treat each video indipendentely, i.e. every video has the
    # same weight.
    average: str = "micro",
    dilation: int | None = None,
):
    wandb.init(
        name=exp_name,
        entity="lparolari",
        project="polypsense-fr",
        dir=os.path.join(os.getcwd(), "wandb_logs"),
        config={
            "data_path": data_path,
            "output_path": output_path,
            "encoder_type": encoder_type,
            "encoder_ckpt": encoder_ckpt,
            "clustering_type": clustering_type,
            "clustering_hparams": clustering_hparams,
            "tracklet_aggregation_method": tracklet_aggregation_method,
            "split": split,
            "video_names": video_names,
            "target_fpr": target_fpr,
            "average": average,
            "dilation": dilation,
        },
    )

    run_id = random.randint(0, 1000000)
    print("run_id", run_id)

    data_path = pathlib.Path(data_path)
    output_dir = pathlib.Path(output_path) / str(run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder(encoder_type, encoder_ckpt, **encoder_kwargs)
    clustering = get_clustering(clustering_type, clustering_hparams)
    get_features = get_features_fn(encoder_type)

    video_info = get_info(data_path / "video_info.csv")
    if video_names is None:
        video_names = get_split(split, video_info)
    video_info = [v for v in video_info if v.unique_video_name in video_names]
    assert len(video_info) == len(video_names)
    wandb.config.update({"video_names": video_names}, allow_val_change=True)
    print(video_names)

    tps_list = []
    fps_list = []
    fns_list = []
    tns_list = []
    recalls_list = []
    precisions_list = []
    fprs_list = []
    tprs_list = []
    preds_list = []
    tracklets_list = []
    entities_list = []

    for v in video_info:
        video_name = v.unique_video_name

        ds = get_ds(video_name, data_path)

        frames_dir = data_path / f"{video_name}_frames"

        samples_features = get_features(
            ds,
            encoder,
            frames_dir=frames_dir,
            agg_method=tracklet_aggregation_method,
            dilation=dilation,
        )
        samples_labels = torch.tensor(get_labels(ds))
        gaps = get_gaps(ds, v.num_frames)

        n = len(ds)
        n_entities = len(samples_labels.unique())

        tracklets_list.append(n)
        entities_list.append(n_entities)

        print("n =", n)
        print("n_entities =", n_entities)

        targets = get_targets(samples_labels)
        scores = get_scores(samples_features)

        print(targets)
        print(scores)

        show_heatmap(scores, output_dir / f"scores_{video_name}.png")
        show_heatmap(targets, output_dir / f"targets_{video_name}.png")

        parameters_space = clustering.parametrize()
        p = len(parameters_space)

        # feats has 2 channels: scores and gaps
        feats = torch.stack([scores, gaps], dim=0)  # [2, n, n]

        preds = torch.stack(
            [
                clustering.fit_predict(feats, parameters)
                for parameters in parameters_space
            ]
        )

        print(preds[0])
        show_heatmap(preds[0], output_dir / f"preds_0_{video_name}.png")

        preds_list.append(preds)

        targets = targets.reshape(-1, n, n)  # [1, g, g]

        tps = ((preds == 1) & (targets == 1)).float().view(p, -1).sum(-1)  # [p]
        fps = ((preds == 1) & (targets == 0)).float().view(p, -1).sum(-1)  # [p]
        fns = ((preds == 0) & (targets == 1)).float().view(p, -1).sum(-1)  # [p]
        tns = ((preds == 0) & (targets == 0)).float().view(p, -1).sum(-1)  # [p]

        if v.num_lesions >= 2:
            tps_list.append(tps)
            fps_list.append(fps)
            fns_list.append(fns)
            tns_list.append(tns)

            recalls = tps / (tps + fns)
            precisions = tps / (tps + fps)
            fprs = fps / (fps + tns)  # https://en.wikipedia.org/wiki/False_positive_rate  # fmt: skip
            fprs = fprs.nan_to_num(nan=0.0)  # fixes the case when tps + fns = 0 (i.e. single polyp video)  # fmt: skip
            tprs = recalls

            recalls_list.append(recalls)
            precisions_list.append(precisions)
            fprs_list.append(fprs)
            tprs_list.append(tprs)

            print(f"{video_name} recalls", tps / (tps + fns))
            print(f"{video_name} precisions", tps / (tps + fps))
            print(f"{video_name} fprs", fps / (fps + tns))

        print(f"{video_name} initial fr", n / n_entities)
        print(f"{video_name} fr preds0", len(associate(preds[0])) / n_entities)

        wandb.log(
            {
                "scores": wandb.Image(str(output_dir / f"scores_{video_name}.png")),
                "targets": wandb.Image(str(output_dir / f"targets_{video_name}.png")),
                "preds_0": wandb.Image(str(output_dir / f"preds_0_{video_name}.png")),
            }
        )

    if average == "micro":
        tps = torch.stack(tps_list).sum(0)  # [v, p] -> [p]
        fps = torch.stack(fps_list).sum(0)  # [v, p] -> [p]
        fns = torch.stack(fns_list).sum(0)  # [v, p] -> [p]
        tns = torch.stack(tns_list).sum(0)  # [v, p] -> [p]

        recalls = tps / (tps + fns)
        precisions = tps / (tps + fps)
        fprs = fps / (fps + tns)  # https://en.wikipedia.org/wiki/False_positive_rate
        tprs = recalls

    if average == "macro":
        recalls = torch.stack(recalls_list).mean(0)
        precisions = torch.stack(precisions_list).mean(0)
        fprs = torch.stack(fprs_list).mean(0)
        tprs = torch.stack(tprs_list).mean(0)

    def sort_and_interpolate(x, y):
        sorted_indices = torch.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        x_interp = np.linspace(0, 1, num=1000)
        y_interp = np.interp(x_interp, x_sorted.numpy(), y_sorted.numpy())

        return torch.tensor(x_interp), torch.tensor(y_interp)

    def auc(x, y):
        return torch.trapz(y, x)

    fprs_roc, tprs_roc = sort_and_interpolate(fprs, tprs)
    recalls_roc, precisions_roc = sort_and_interpolate(recalls, precisions)

    auroc = auc(fprs_roc, tprs_roc)
    auprc = auc(recalls_roc, precisions_roc)

    show_p_r(recalls, precisions, output_dir / "p_r.png")
    show_roc_curve(fprs_roc, tprs_roc, output_dir / "roc_curve.png")
    show_pr_curve(recalls_roc, precisions_roc, output_dir / "pr_curve.png")

    print("auroc", auroc)
    print("auprc", auprc)

    best_i = select_best_parameters(fprs, target_fpr=target_fpr)
    best_parameters = clustering.parametrize()[best_i]
    operating_fpr = fprs[best_i]

    print("best_i", best_i)
    print("best_parameters", best_parameters)
    print("operating_fpr", operating_fpr)

    preds_per_video = [preds[best_i] for preds in preds_list]
    fragments_per_video = [associate(pred) for pred in preds_per_video]

    fr_per_video = [
        len(f) / n_entities for f, n_entities in zip(fragments_per_video, entities_list)
    ]
    initial_fr_per_video = [
        n / n_entities for n, n_entities in zip(tracklets_list, entities_list)
    ]

    fr = torch.tensor(fr_per_video).mean()
    fr_std = torch.tensor(fr_per_video).std()
    initial_fr = torch.tensor(initial_fr_per_video).mean()

    print("initial_fr_per_video", initial_fr_per_video)
    print("fr_per_video", fr_per_video)
    print("initial fr", initial_fr)
    print("fr", fr)
    print("fr_std", fr_std)
    print("run_id", run_id)

    wandb.log(
        {
            "fr": fr,
            "fr_std": fr_std,
            "auroc": auroc.item(),
            "auprc": auprc.item(),
            "fpr": operating_fpr.item(),
            "best_parameters": best_parameters,
            "p_r": wandb.Image(str(output_dir / "p_r.png")),
            "roc_curve": wandb.Image(str(output_dir / "roc_curve.png")),
            "pr_curve": wandb.Image(str(output_dir / "pr_curve.png")),
        }
    )


def get_encoder(encoder_type, encoder_ckpt, **kwargs):
    encoder = get_backbone(encoder_type, encoder_ckpt, **kwargs)
    encoder.load()
    return encoder


def get_split(split, video_info):
    if split == "val":
        return [
            v.unique_video_name
            for v in video_info
            if v.num_lesions >= 1 and 9 <= int(v.unique_video_name.split("-")[1]) <= 10
        ]
    if split == "test":
        return [
            v.unique_video_name
            for v in video_info
            if v.num_lesions >= 1 and int(v.unique_video_name.split("-")[1]) >= 11
        ]
    raise ValueError(f"Unknown split '{split}'")


def get_ds(video_name, data_path):
    annotations_dir = data_path / f"{video_name}_annotations"
    annotations_file_list = load_files_list(annotations_dir)
    annotations_list = [parsevocfile(f) for f in annotations_file_list]

    polyps = get_polyps(annotations_list)
    return FragmentationRateDataset(polyps)


def get_single_frame_features(ds, model, *, frames_dir, agg_method="mean", **kwargs):
    tracklets_feat = []

    for i in tqdm(range(len(ds))):
        tracklet = ds[i]

        tracklet_feats = []

        for polyp_id in tracklet:
            polyp_ann = ds.polyp(polyp_id)
            image_path = frames_dir / polyp_ann["frame_name"]
            bbox = polyp_ann["bbox"]  # xywh
            bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

            # TODO: since different models may need to crop to the bounding box
            # or either use the whole image we forward to the model itself the
            # bounding box such that it can decide what to do. This however is a
            # sort of "dataloader" problem and should be done in the preparation
            # of data not by the model itself
            tracklet_feat = model.forward(image_path, bbox_xyxy)
            tracklet_feats.append(tracklet_feat)

        if agg_method == "mean":
            tracklet_repr = torch.stack(tracklet_feats).mean(dim=0)
        elif agg_method == "max":
            tracklet_repr = torch.stack(tracklet_feats).max(dim=0).values
        elif agg_method == "min":
            tracklet_repr = torch.stack(tracklet_feats).min(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation method {agg_method}")

        tracklets_feat.append(tracklet_repr)

    return torch.cat(tracklets_feat).cpu()


def get_multi_view_features(ds, model, *, frames_dir, dilation=1, **kwargs):
    tracklets_feat = []

    for i in tqdm(range(len(ds))):
        tracklet = ds[i]

        images = []
        bboxes = []

        tracklet_selected = tracklet[::dilation]

        # the original tracklet must be at least 32 frames long to use dilation
        # otherwise performance may be affected by the lack of data
        if len(tracklet_selected) < 8:
            tracklet_selected = tracklet

        for polyp_id in tracklet_selected:
            polyp_ann = ds.polyp(polyp_id)
            image_path = frames_dir / polyp_ann["frame_name"]
            bbox = polyp_ann["bbox"]
            bbox_xyxy = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

            images.append(image_path)
            bboxes.append(bbox_xyxy)

        tracklet_feat = model.forward(images, bboxes)

        tracklets_feat.append(tracklet_feat)

    return torch.cat(tracklets_feat).cpu()


def get_features_fn(encoder_type):
    return {
        "simclr": get_single_frame_features,
        "rtdetr": get_single_frame_features,
        "mve": get_multi_view_features,
    }[encoder_type]


def get_gaps(ds, num_frames):
    n = len(ds)

    gaps = torch.zeros(n, n)

    for i in range(n):
        for j in range(n):
            gaps[i, j] = ds.gap(i, j)

    # normalize gaps between 0 and 1
    gaps = gaps / num_frames

    # since tracklets can overlap, negative gaps are possible we set them to
    # +inf because no re-identification is possible between overlapping
    # tracklets
    gaps[gaps < 0] = float("inf")

    return gaps.cpu()


def get_scores(x):
    """
    Return similarity scores between all pairs of samples.

    Args:
        x: [n, d] features

    Returns:
        scores: [n, n] similarity matrix
    """
    # get pairwise distances
    dists = torch.cdist(x, x, p=2)

    # normalize distances between 0 and 1
    d_min = dists.min(-1, keepdim=True).values
    d_max = dists.max(-1, keepdim=True).values
    distances = (dists - d_min) / (d_max - d_min)

    # the bigger the better
    scores = 1 - distances  # [g, g]

    return scores


def get_targets(y):
    """
    Return a boolean matrix indicating whether pairs of samples have the same
    label.

    Args:
        y: [n] labels

    Returns:
        targets: [n, n] boolean matrix
    """
    return y.unsqueeze(1) == y.unsqueeze(0)


def select_best_parameters(fprs, target_fpr):
    diff = torch.abs(fprs - target_fpr)
    return torch.argmin(diff).item()


def show_heatmap(x, out_path):
    plt.imshow(x, cmap="viridis")
    plt.savefig(out_path)
    plt.close()


def show_p_r(recalls, precisions, out_path):
    x = torch.arange(recalls.size(0))
    plt.plot(x, recalls, label="Recall")
    plt.plot(x, precisions, label="Precision")
    plt.xlabel("Param group")
    plt.legend()
    plt.savefig(out_path)
    plt.close()


def show_roc_curve(fprs, tprs, out_path):
    plt.plot(fprs, tprs)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.savefig(out_path)
    plt.close()


def show_pr_curve(recalls, precisions, out_path):
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(out_path)
    plt.close()


def auc(x, y, reorder=False):
    # Source: https://pytorch.org/torcheval/main/generated/torcheval.metrics.functional.auc.html

    if x.numel() == 0 or y.numel() == 0:
        return torch.tensor([])

    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)

    if reorder:
        x, x_idx = torch.sort(x, dim=1, stable=True)
        y = y.gather(1, x_idx)

    return torch.trapz(y, x)


# -----------------------------------------------------------------------------
# RUNTIME
# -----------------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument(
        "--data_path", type=str, default="data/real-colon/raw", required=True
    )
    parser.add_argument("--output_path", type=str, default="tmp", required=True)
    parser.add_argument(
        "--encoder_type", type=str, required=True, choices=["simclr", "rtdetr", "mve"]
    )
    parser.add_argument("--encoder_ckpt", type=str, required=True)
    parser.add_argument(
        "--clustering_type",
        type=str,
        required=True,
        choices=[
            "threshold",
            "agglomerative",
            "dbscan",
            "affinity_propagation",
            "temporal",
        ],
    )
    parser.add_argument("--encoder_kwargs", type=json.loads, default={})
    parser.add_argument("--clustering_hparams", type=json.loads, default=None)
    parser.add_argument(
        "--tracklet_aggregation_method",
        type=str,
        default=None,
        choices=["mean", "max", "min"],
    )
    parser.add_argument("--split", type=str, required=None, choices=["val", "test"])
    parser.add_argument("--video_names", type=str, nargs="+", default=None)
    parser.add_argument("--target_fpr", type=float, default=0.05)
    parser.add_argument(
        "--metric_average", type=str, default="micro", choices=["micro", "macro"]
    )
    parser.add_argument("--dilation", type=int, default=None)

    return parser


def main():
    args = get_parser().parse_args()
    _validate_args(args)

    run(
        exp_name=args.exp_name,
        data_path=args.data_path,
        output_path=args.output_path,
        encoder_type=args.encoder_type,
        encoder_ckpt=args.encoder_ckpt,
        encoder_kwargs=args.encoder_kwargs,
        clustering_type=args.clustering_type,
        clustering_hparams=args.clustering_hparams,
        tracklet_aggregation_method=args.tracklet_aggregation_method,
        split=args.split,
        video_names=args.video_names,
        target_fpr=args.target_fpr,
        average=args.metric_average,
        dilation=args.dilation,
    )


def _validate_args(args):
    if (not args.split and not args.video_names) or (args.split and args.video_names):
        raise ValueError("One of `split` or `video_names` must be provided")

    if args.encoder_type == "mve":
        if not args.dilation:
            raise ValueError("Dilation must be provided for MVE encoder")

        if args.dilation < 1:
            raise ValueError("Dilation must be greater than 0")


if __name__ == "__main__":
    main()
