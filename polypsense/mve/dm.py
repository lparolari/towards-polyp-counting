import pathlib

import lightning as L
import torch
from torchvision.transforms import v2

from polypsense.mve.data import PolypDataset, TrackletDataset


class SequenceTemporalDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root,
        s,
        im_size,
        aug_vflip,
        aug_hflip,
        aug_affine,
        aug_colorjitter,
        batch_size,
        num_workers,
        train_ratio,
        alpha,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.s = s
        self.im_size = im_size
        self.aug_vflip = aug_vflip
        self.aug_hflip = aug_hflip
        self.aug_affine = aug_affine
        self.aug_colorjitter = aug_colorjitter
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.alpha = alpha

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_ds("train")
            self.val_dataset = self._make_ds("val")

        if stage == "test" or stage is None:
            self.test_dataset = self._make_ds("test")

    def train_dataloader(self):
        p = self.train_ratio
        n = len(self.train_dataset)
        # since TrackletDataset.__getitem__ randomly samples items it is quite
        # useless to randperm the indices and setting the generator seed
        sampler = torch.utils.data.SubsetRandomSampler(
            indices=torch.randperm(n)[: int(p * n)],
            generator=torch.Generator().manual_seed(42),
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            batch_size=1,  # since items in batch are not independent, we build batches in the dataset
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )

    def val_dataloader(self):
        return self._get_eval_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def _get_eval_dataloader(self, ds):
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=collate_fn,
            batch_size=1,  # since items in batch are not independent, we build batches in the dataset
            num_workers=self.num_workers,
            pin_memory=True,
            # We need to shuffle the validation set, but we fix the sampler to
            # the same random sequence in order to promote reproducibility.
            sampler=torch.utils.data.RandomSampler(
                ds, generator=torch.Generator().manual_seed(42)
            ),
        )

    def _make_ds(self, split):
        # TODO: should be moved in a function
        import os
        if os.path.exists(f"instances_{split}.trackleter.pkl"):
            import pickle
            print("Load saved trackleter")
            with open(f"instances_{split}.trackleter.pkl", "rb") as f:
                trackleter = pickle.load(f)
        else:
            trackleter = None
            
        img_folder = pathlib.Path(self.dataset_root) / "images"
        ann_file = (
            pathlib.Path(self.dataset_root) / "annotations" / f"instances_{split}.json"
        )

        ds = PolypDataset.from_instances(img_folder, ann_file)
        ds = TrackletDataset(
            ds,
            s=self.s,
            batch_size=self.batch_size,
            transforms=self._get_transforms(split),
            trackleter=trackleter,
            alpha=self._get_alpha(split),
        )

        return ds

    def _get_transforms(self, split):
        transforms_fn = {
            "train": lambda: self.get_train_transforms(
                self.im_size,
                self.aug_vflip,
                self.aug_hflip,
                self.aug_affine,
                self.aug_colorjitter,
            ),
            "val": lambda: self.get_test_transforms(self.im_size),
            "test": lambda: self.get_test_transforms(self.im_size),
        }[split]

        return transforms_fn()

    def _get_alpha(self, split):
        return {
            "train": self.alpha,
            "val": 0.5,
            "test": 0.5,
        }[split]

    @staticmethod
    def get_train_transforms(
        im_size,
        aug_vflip,
        aug_hflip,
        aug_affine,
        aug_colorjitter,
    ):
        return v2.Compose(
            [
                v2.Resize(size=[im_size] * 2),
            ]
            + ([v2.RandomHorizontalFlip()] if aug_hflip else [])
            + ([v2.RandomVerticalFlip()] if aug_vflip else [])
            + (
                [v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))]
                if aug_affine
                else []
            )
            + (
                [v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)]
                if aug_colorjitter
                else []
            )
            + [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    @staticmethod
    def get_test_transforms(im_size):
        return v2.Compose(
            [
                v2.Resize(size=[im_size] * 2),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )


def collate_fn(batch):
    return batch[0]
