import pathlib

import lightning as L
import torch
from torchvision.transforms import v2

from polypsense.sfe.data import (
    AugmentationContrastiveDataset,
    Cifar10Dataset,
    RealColonPolypsDataset,
    TemporalContrastiveDataset,
)
from polypsense.sfe.view import (
    GaussianTemporalViewGenerator,
    UniformTemporalViewGenerator,
)


def collate_fn(batch):
    # batch is a list of `batch_size` tuples (view1, view2)
    # each view is an image, i.e. a tensor of shape [c, h, w]
    batch = torch.utils.data.dataloader.default_collate(batch)  # [v, b, c, h, w]
    return torch.cat(batch, dim=0)  # [v*b, c, h, w]


def make_realcolon(data_root, split):
    img_folder = pathlib.Path(data_root) / "images"
    ann_file = pathlib.Path(data_root) / "annotations" / f"instances_{split}.json"
    return RealColonPolypsDataset.from_instances(img_folder, ann_file)


def make_cifar10(data_root, split):
    return Cifar10Dataset(data_root, train=split == "train")


class TemporalDataModule(L.LightningDataModule):
    _supported_datasets = ["realcolon"]

    def __init__(self, args):
        super().__init__()

        if args.dataset_name not in self._supported_datasets:
            raise ValueError(
                f"Unsupported dataset '{args.dataset_name}' for temporal augmentation"
            )

        self.args = args

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_ds("train")
            self.val_dataset = self._make_ds("val")

        if stage == "test" or stage is None:
            self.test_dataset = self._make_ds("test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return self._get_eval_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    def _get_eval_dataloader(self, ds):
        return torch.utils.data.DataLoader(
            ds,
            collate_fn=collate_fn,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            # We need to shuffle the validation set, but we fix the sampler to
            # the same random sequence in order to promote reproducibility.
            sampler=torch.utils.data.RandomSampler(
                ds, generator=torch.Generator().manual_seed(42)
            ),
        )

    def _make_ds(self, split):
        base_ds = self._get_base_ds(split)
        view_generator = self._get_view_generator()
        transforms = self._get_transforms(split)

        ds = TemporalContrastiveDataset(
            base_ds,
            view_generator=view_generator,
            transforms=transforms,
        )

        return ds

    def _get_base_ds(self, split):
        baseds_fn = {
            "realcolon": make_realcolon,
        }[self.args.dataset_name]

        return baseds_fn(self.args.dataset_root, split)

    def _get_transforms(self, split):
        transforms_fn = {
            "train": lambda: self.get_train_transforms(
                self.args.im_size,
                self.args.aug_vflip,
                self.args.aug_hflip,
                self.args.aug_affine,
                self.args.aug_colorjitter,
            ),
            "val": lambda: self.get_test_transforms(self.args.im_size),
            "test": lambda: self.get_test_transforms(self.args.im_size),
        }[split]

        return transforms_fn()

    def _get_view_generator(self):
        generators_fn = {
            "gaussian": lambda: GaussianTemporalViewGenerator(
                std=self.args.view_generator_gaussian_std
            ),
            "uniform": lambda: UniformTemporalViewGenerator(
                window=self.args.view_generator_uniform_window
            ),
        }

        return generators_fn[self.args.view_generator]()

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


class AugmentationDataModule(L.LightningDataModule):
    _supported_datasets = ["realcolon", "cifar10"]

    def __init__(self, args):
        super().__init__()

        if args.dataset_name not in self._supported_datasets:
            raise ValueError(
                f"Unsupported dataset '{args.dataset_name}' for plain augmentation"
            )

        self.args = args

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._make_ds("train")
            self.val_dataset = self._make_ds("val")
        if stage == "test" or stage is None:
            self.test_dataset = self._make_ds("test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=collate_fn,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=collate_fn,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            collate_fn=collate_fn,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    def _make_ds(self, split):
        base_ds = self._get_base_ds(split)
        transform = self.get_transform(self.args.im_size)
        return AugmentationContrastiveDataset(base_ds, transform=transform)

    def _get_base_ds(self, split):
        baseds_fn = {
            "realcolon": make_realcolon,
            "cifar10": make_cifar10,
        }[self.args.dataset_name]

        return baseds_fn(self.args.dataset_root, split)

    @staticmethod
    def get_transform(im_size):
        from torchvision.transforms import v2

        # Adapted from https://github.com/sthalles/SimCLR/blob/1848fc934ad844ae630e6c452300433fe99acfd9/data_aug/contrastive_learning_dataset.py
        s = 1  # this is always set to 1 in the reference implementation

        ks = int(0.1 * im_size)

        color_jitter = v2.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = v2.Compose(
            [
                v2.RandomResizedCrop(size=im_size),
                v2.RandomHorizontalFlip(),
                v2.RandomApply([color_jitter], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.GaussianBlur(
                    kernel_size=ks if ks % 2 else ks + 1
                ),  # needs odd kernel size
                # ToTensor() has been deprected, we use the ToImage + ToDtype
                # following documentation
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        return data_transforms
