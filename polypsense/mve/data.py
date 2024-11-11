import random

import torch

from polypsense.mve.sequencer import Sequencer
from polypsense.mve.trackleter import Trackleter
from polypsense.sfe.data import CocoDataset

SequenceId = str | int
SampleId = int


class BaseDataset:
    def id(self, index):
        raise NotImplementedError

    def img(self, id):
        raise NotImplementedError

    def img_ann(self, id):
        raise NotImplementedError

    def tgt_ann(self, id):
        raise NotImplementedError


class PolypDataset(torch.utils.data.Dataset, BaseDataset):
    """
    Each item in the dataset is an image cropped around the annotation.
    """

    def __init__(self, ds: CocoDataset):
        self.ds = ds
        self._build_ids()

    def _build_ids(self):
        self.targets = [
            t for i in range(len(self.ds)) for t in self.ds.tgt_ann(self.ds.id(i))
        ]
        # keeping only samples with targets
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
        return PolypDataset(CocoDataset(root, ann_file))


class TrackletDataset(torch.utils.data.Dataset):
    """
    Build a batch of pairs where each pair is composed by samples from two
    different tracklets.
    """

    def __init__(
        self,
        ds: BaseDataset,
        *,
        s=8,
        batch_size=16,
        transforms=None,
        trackleter=None,
        alpha=1.0,
    ):
        """
        Args:
            ds: dataset with instances
            s: length of the sequence to create
            batch_size: number of positive pairs to return
            transforms: optional transforms to apply to the images
        """
        self.ds = ds
        self.sequencer = Sequencer(ds)
        self.trackleter = trackleter or Trackleter(ds)
        self.s = s
        self.batch_size = batch_size
        self.transforms = transforms
        self.alpha = alpha

    def __len__(self):
        return len(self.trackleter)

    def __getitem__(self, idx):
        # Sequences are always different, thus we always sample just two pair of
        # tracklets per sequence. Note that this does not take into account
        # imbalance between the number of tracklet in sequences.

        sampled_sequences = self._sample_sequence_ids(self.batch_size)

        batch_a = []
        batch_b = []

        for sequence in sampled_sequences:
            if random.random() <= self.alpha:
                view_a, view_b = self._sample_views_from_same_tracklet(sequence)
            else:
                view_a, view_b = self._sample_views_from_different_tracklets(sequence)

            view_a = self._get_frames(view_a)
            if self.transforms:
                view_a = self.transforms(view_a)
            batch_a.append(view_a)

            view_b = self._get_frames(view_b)
            if self.transforms:
                view_b = self.transforms(view_b)
            batch_b.append(view_b)

        if isinstance(batch_a[0][0], torch.Tensor):
            return torch.cat(
                [
                    torch.stack([torch.stack(frames) for frames in batch_a]),
                    torch.stack([torch.stack(frames) for frames in batch_b]),
                ],
                dim=0,
            )  # [b*2, s, c, h, w]

        return batch_a + batch_b

    def _sample_sequence_ids(self, batch_size):
        return random.sample(self.sequencer.list_sequence_ids(), batch_size)

    def _sample_views_from_same_tracklet(self, sequence_id):
        tracklet_ids = self.trackleter.get_tracklet_ids_by_sequence(
            sequence_id, min_length=30
        )
        tracklet_lengths = [
            len(self.trackleter.get_tracklet(tid)) for tid in tracklet_ids
        ]

        # Sample a tracklet with slightly more probability if it is longer.
        weights = [l**0.5 for l in tracklet_lengths]
        tracklet_id = random.choices(tracklet_ids, weights=weights)[0]

        # We need to sample a pair of views for the same tracklet, each of them
        # should be of `self.s` frames with stride in 1..4. We must try to
        # minimize overlap otherwise it is easy to overfit for the model. Each
        # view might also use a different scaling factor.

        tracklet = self.trackleter.get_tracklet(tracklet_id)
        n = len(tracklet)
        m = int(n / 2)  # num of frames available for each view

        # print(len(tracklet))
        # print("n", n)
        # print("m", m)

        dilations = [1, 2, 3, 4]

        def select_stride():
            # stride must be selected such that we have enough frames for that
            # stride, i.e. self.s * d < m. give more weight to larger strides
            weights = [int(self.s * d < m) * d for d in dilations]
            stride = random.choices(dilations, weights=weights)[0]
            return stride

        def select_first_frame(i, stride):
            # first frame must be selected such that we manage to stay within m frames with given stride
            first_frame_range = list(range(i, i + m - self.s * stride)) or [0]
            return random.choice(first_frame_range)

        stride = select_stride()

        # print("stride", stride)

        view_a_i = select_first_frame(0, stride)
        view_b_i = select_first_frame(view_a_i + self.s * stride, stride)

        # print("view_a_i", view_a_i)
        # print("view_b_i", view_b_i)

        view_a = tracklet[view_a_i : view_a_i + self.s * stride : stride]
        view_b = tracklet[view_b_i : view_b_i + self.s * stride : stride]

        return view_a, view_b

    def _sample_views_from_different_tracklets(self, sequence_id):
        tracklet_ids = self.trackleter.get_tracklet_ids_by_sequence(
            sequence_id, min_length=30
        )
        tracklet_lengths = [
            len(self.trackleter.get_tracklet(tid)) for tid in tracklet_ids
        ]

        # Sample a tracklet with slightly more probability if it is longer.
        weights = [l**0.5 for l in tracklet_lengths]

        tracklet_id_a = random.choices(tracklet_ids, weights=weights)[0]
        tracklet_id_b = random.choices(tracklet_ids, weights=weights)[0]

        for _ in range(1000):
            if tracklet_id_a != tracklet_id_b:
                break
            tracklet_id_b = random.choices(tracklet_ids, weights=weights)[0]
        else:
            raise ValueError("Could not sample different tracklets")

        # print("tracklet_id_a", tracklet_id_a)
        # print("tracklet_id_b", tracklet_id_b)

        tracklet_a = self.trackleter.get_tracklet(tracklet_id_a)
        tracklet_b = self.trackleter.get_tracklet(tracklet_id_b)

        def select_frames(tracklet):
            n = len(tracklet)
            dilations = [1, 2, 3, 4]

            weights = [int(self.s * d < n) for d in dilations]  # 1 or 0
            stride = random.choices(dilations, weights=weights)[0]

            first_frame_range = list(range(0, n - self.s * stride)) or [0]
            first_frame = random.choice(first_frame_range)

            return tracklet[first_frame : first_frame + self.s * stride : stride]

        view_a = select_frames(tracklet_a)
        view_b = select_frames(tracklet_b)

        return view_a, view_b

    def _get_frames(self, frame_ids):
        return [self._get_frame(frame_id) for frame_id in frame_ids]

    def _get_frame(self, frame_id):
        img = self.ds.img(frame_id)

        bbox = self.ds.tgt_ann(frame_id)["bbox"]
        img = self._crop_image(img, bbox)

        return img

    def _crop_image(self, img, bbox):
        x, y, w, h = bbox
        return img.crop((x, y, x + w, y + h))
