import argparse
import os
import pathlib

import torch
from tqdm import tqdm

import wandb
from polypsense.dataset.realcolon import get_info, load_files_list, parsevocfile
from polypsense.reid2.backbone import get_backbone
from polypsense.reid2.dataset import get_polyps
from polypsense.reid2.metric import get_cmc_k_score, get_map_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--ckpt_path", type=str)
    args = parser.parse_args()
    run(args)


def run(args):
    raise NotImplementedError("This script is not working, please implement dataset")
    wandb.init(
        name=args.exp_name,
        entity="lparolari",
        project="polypsense-retrieval",
        dir=os.path.join(os.getcwd(), "wandb_logs"),
        config=args,
    )

    data_path = pathlib.Path(args.data_path)

    video_info = get_info(data_path / "video_info.csv")
    video_info = [v for v in video_info if v.num_lesions > 0]

    model = get_backbone(args.model_type, args.ckpt_path)
    model.load()

    cmc1_cum = torch.tensor(0.0)
    cmc3_cum = torch.tensor(0.0)
    cmc5_cum = torch.tensor(0.0)
    cmc10_cum = torch.tensor(0.0)
    mAP_cum = torch.tensor(0.0)

    n = 0

    for video in video_info:
        annotations_dir = data_path / f"{video.unique_video_name}_annotations"
        annotations_file_list = load_files_list(annotations_dir)
        annotations_list = [parsevocfile(f) for f in annotations_file_list]

        frames_dir = data_path / f"{video.unique_video_name}_frames"

        polyps = get_polyps(annotations_list)
        ds = RealColonTrackletsDataset(polyps)

        assert len(ds) != 0, "Video without polyp entities"

        n += 1

        samples_features = get_features(ds, model, frames_dir)
        samples_labels = get_labels(ds)

        cmc1, cmc3, cmc5, cmc10, mAP = eval(samples_features, samples_labels)

        cmc1_cum += cmc1
        cmc3_cum += cmc3
        cmc5_cum += cmc5
        cmc10_cum += cmc10
        mAP_cum += mAP

    cmc1 = (cmc1_cum / n).item()
    cmc3 = (cmc3_cum / n).item()
    cmc5 = (cmc5_cum / n).item()
    cmc10 = (cmc10_cum / n).item()
    mAP = (mAP_cum / n).item()

    print(f"cmc@1 {cmc1:.3f}")
    print(f"cmc@3 {cmc3:.3f}")
    print(f"cmc@5 {cmc5:.3f}")
    print(f"cmc@10 {cmc10:.3f}")
    print(f"mAP {mAP:.3f}")

    wandb.log(
        {
            "cmc1": cmc1,
            "cmc3": cmc3,
            "cmc5": cmc5,
            "cmc10": cmc10,
            "mAP": mAP,
        }
    )


def get_labels(ds):

    def _get_sequence(ds, tracklet_idx):
        tracklet = ds[tracklet_idx]
        all_sequences = set([ds.polyp(polyp_id)["sequence"] for polyp_id in tracklet])

        if len(all_sequences) != 1:
            raise ValueError("Tracklet contains polyps from different sequences")

        return list(all_sequences)[0]

    all_sequences = list(set([_get_sequence(ds, i) for i in range(len(ds))]))
    all_sequences.sort()

    samples_idx = list(range(len(ds)))
    samples_sequences = [_get_sequence(ds, i) for i in samples_idx]
    samples_labels = [all_sequences.index(seq) for seq in samples_sequences]

    return samples_labels


def get_features(ds, model, frames_dir, aggregation_method="mean"):
    tracklets_feat = []

    for i in tqdm(range(len(ds))):
        tracklet = ds[i]

        tracklet_feats = []

        for polyp_id in tracklet:
            polyp_ann = ds.polyp(polyp_id)
            image_path = frames_dir / polyp_ann["frame_name"]
            bbox = polyp_ann["bbox"]

            # TODO: since different models may need to crop to the bounding box
            # or either use the whole image we forward to the model itself the
            # bounding box such that it can decide what to do. This however is a
            # sort of "dataloader" problem and should be done in the preparation
            # of data not by the model itself
            tracklet_feat = model.forward(image_path, bbox)
            tracklet_feats.append(tracklet_feat)

        if aggregation_method == "mean":
            tracklet_repr = torch.stack(tracklet_feats).mean(dim=0)
        elif aggregation_method == "max":
            tracklet_repr = torch.stack(tracklet_feats).max(dim=0).values
        elif aggregation_method == "min":
            tracklet_repr = torch.stack(tracklet_feats).min(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation method {aggregation_method}")

        tracklets_feat.append(tracklet_repr)

    return torch.cat(tracklets_feat).cpu()


def eval(features, labels):

    cmc1_cum = torch.tensor(0.0)
    cmc3_cum = torch.tensor(0.0)
    cmc5_cum = torch.tensor(0.0)
    cmc10_cum = torch.tensor(0.0)
    mAP_cum = torch.tensor(0.0)

    n = len(labels)

    for i in range(n):
        query_idx = i
        gallery_idx = [i for i in range(n) if i != query_idx]

        query_labels = torch.tensor(labels[query_idx]).unsqueeze(0)

        query_feat = features[query_idx].unsqueeze(0)
        gallery_feat = features[gallery_idx]

        distances = torch.cdist(query_feat, gallery_feat, p=2)

        gallery_labels = torch.tensor(labels)[gallery_idx]

        predictions = gallery_labels[torch.argsort(distances)]

        cmc1_cum += get_cmc_k_score(query_labels, predictions, 1)
        cmc3_cum += get_cmc_k_score(query_labels, predictions, 3)
        cmc5_cum += get_cmc_k_score(query_labels, predictions, 5)
        cmc10_cum += get_cmc_k_score(query_labels, predictions, 10)
        mAP_cum += get_map_score(query_labels, predictions, gallery_labels)

    cmc1 = (cmc1_cum / n).item()
    cmc3 = (cmc3_cum / n).item()
    cmc5 = (cmc5_cum / n).item()
    cmc10 = (cmc10_cum / n).item()
    mAP = (mAP_cum / n).item()

    return cmc1, cmc3, cmc5, cmc10, mAP


if __name__ == "__main__":
    main()
