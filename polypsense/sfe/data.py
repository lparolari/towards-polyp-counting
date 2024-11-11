import contextlib
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms.v2

from polypsense.dataset.realcolon import get_frame_id


class Cifar10Dataset(torchvision.datasets.CIFAR10):
    pass


class Stl10Dataset(torchvision.datasets.STL10):
    pass


class AugmentationContrastiveDataset(torch.utils.data.Dataset):
    """
    Implements a dataset for contrastive learning with augmentations as
    described in the SimCLR paper.
    """

    def __init__(self, ds, transform):
        if not callable(transform):
            raise ValueError(f"Transform must be a callable, found '{transform}'")

        self.ds = ds
        self.transform = transform

    def __getitem__(self, index):
        img, _ = self.ds[index]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2

    def __len__(self):
        return len(self.ds)


class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(
        self, root, annFile, transform=None, target_transform=None, transforms=None
    ):
        # suppress pycocotools prints
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                super().__init__(root, annFile, transform, target_transform, transforms)

    def id(self, index):
        return self.ids[index]

    def img(self, id):
        return self._load_image(id)

    def img_ann(self, id):
        return self.coco.loadImgs(id)[0]

    def tgt_ann(self, id):
        return self._load_target(id)


class RealColonPolypsDataset(
    torch.utils.data.Dataset
):  # TODO: may be renamed to resemble the single target property
    """
    Represents each polyp as a separate item in the dataset.
    """

    def __init__(self, ds: CocoDataset):
        self.ds = ds

        self._prepare()
        # self._filter_by_bbox_size(area_range=(0.03, 0.9))

    def _prepare(self):
        self.targets = [
            t for i in range(len(self.ds)) for t in self.ds.tgt_ann(self.ds.id(i))
        ]
        self.ids = [i for i in range(len(self.targets))]

    def _filter_by_bbox_size(self, area_range):
        """
        Filters out targets based on the area of the bounding box.
        """
        min_area, max_area = area_range

        new_targets = []
        new_ids = []

        for i in range(len(self)):
            tgt = self.tgt_ann(self.id(i))
            img = self.img_ann(self.id(i))
            im_w = img["width"]
            im_h = img["height"]

            area = tgt["area"] / (im_w * im_h)  # normalize by image size

            if min_area <= area <= max_area:
                new_targets.append(tgt)
                new_ids.append(i)

        self.targets = new_targets
        self.ids = [i for i in range(len(self.targets))]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.id(index)
        return self.img(id), self.tgt_ann(id)

    def id(self, index):
        return self.ids[index]

    def img(self, id):
        target = self.targets[id]
        image_id = target["image_id"]
        return self.ds.img(image_id)

    def img_ann(self, id):
        target = self.targets[id]
        image_id = target["image_id"]
        return self.ds.img_ann(image_id)

    def tgt_ann(self, id):
        return self.targets[id]

    @staticmethod
    def from_instances(root, ann_file):
        return RealColonPolypsDataset(CocoDataset(root, ann_file))


class TemporalContrastiveDataset(torch.utils.data.Dataset):
    """
    Implements a dataset for contrastive learning with SimCLR framework as
    described in https://arxiv.org/pdf/2306.08591 (Sec 2.1).

    Applies "temporal" augmentation for contrastive learning leveraging data
    sequences identified by a `sequence_key`.
    """

    def __init__(self, ds, view_generator, transforms=None, sequence_key="sequence_id"):
        self.ds = ds
        self.view_generator = view_generator
        self.transforms = transforms
        self.sequence_key = sequence_key

        self._build_sequence_index()

    def __len__(self):
        return len(self.index2sequence)

    def __getitem__(self, idx):
        """
        Returns a pair of images that belongs to the same sequence. Images are
        cropped to the bounding box with the object of interest.
        """
        idx2 = self.sample_view(idx)

        view1 = self.get_view(idx)
        view2 = self.get_view(idx2)

        return view1, view2

    def sample_view(self, view1_idx):
        view2_candidates = self._get_view_candidates(view1_idx)
        view2_idx = self.view_generator.sample(view1_idx, view2_candidates)
        return view2_idx

    def get_view(self, idx):
        """
        Return a cropped image of the object of interest for the given index,
        applies transforms.
        """
        view = self.ds.img(self.ds.id(idx))

        view = self._crop_image(view, self.ds.tgt_ann(self.ds.id(idx))["bbox"])

        if self.transforms:
            view = self.transforms(view)

        return view

    def _get_view_candidates(self, view1_idx):
        view1_sequence = self.index2sequence[view1_idx]

        # get candidates views for given sequence
        view2_candidates = np.array(self.sequence2index[view1_sequence])

        # ensure candidates are sorted by temporal oreder, through frame
        # identifiers
        view2_frameids = np.array(
            [
                get_frame_id(self.ds.img_ann(self.ds.id(i))["file_name"])
                for i in view2_candidates
            ]
        )

        # do the sorting
        view2_candidates = view2_candidates[np.argsort(view2_frameids)]

        return view2_candidates

    def _crop_image(self, img, bbox):
        """
        Crop image to the bounding box.
        """
        x, y, w, h = bbox
        return img.crop((x, y, x + w, y + h))

    def _build_sequence_index(self):
        """
        Builds two indices:
        * sequence -> list of indices, with the list of indices belonging to given sequence
        * index -> sequence, with the sequence id for each given index
        """

        # NOTE: this code may be moved into the RealColonPolypsDataset class
        # such that it becomes the object to query for information like the
        # available sequences and the ids of samples belonging to a given
        # sequence.
        # The interface could be something like:
        #   sequences(self) -> List[SequenceId]
        #   sequence(self, sequence_id) -> List[SampleId]
        # We do not need to move it now, if needed it will become clear in the
        # future. This will allow us also to explicitly test dataset properties.

        sequence2index = {}
        index2sequence = {}

        for i in range(len(self.ds)):
            tgt_ann = self.ds.tgt_ann(self.ds.id(i))
            sequence_id = tgt_ann[self.sequence_key]
            if sequence_id not in sequence2index:
                sequence2index[sequence_id] = []

            sequence2index[sequence_id].append(i)
            index2sequence[i] = sequence_id

        assert len(index2sequence) == len(
            [v for vs in sequence2index.values() for v in vs]
        )

        self.sequence2index = sequence2index
        self.index2sequence = index2sequence
