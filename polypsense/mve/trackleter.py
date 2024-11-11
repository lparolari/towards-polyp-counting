from functools import lru_cache

from tqdm import tqdm

from polypsense.dataset.realcolon import get_frame_id

Tracklet = list[int]


class Trackleter:
    """
    Build a list of tracklets out of a collection of items. Tracklets are
    separated based on temporal and spatial constraints.
    """

    def __init__(self, ds, separation_thresh=1, iou_thresh=0.1, pbar=False):
        self.ds = ds
        self.separation_thresh = separation_thresh
        self.iou_thresh = iou_thresh
        self.pbar = pbar

        self._tracklets = {}
        self._tracklets_meta = {}
        self._tracklet_ids = []
        self._tracklet_counter = 0

        self._build_tracklets()

    def __getitem__(self, idx) -> Tracklet:
        return self._get(self._tracklet_ids[idx])

    def __len__(self):
        return len(self._tracklets)

    def get_tracklet_ids(self):
        return self._tracklet_ids

    def get_tracklet_ids_by_sequence(self, sequence_id, min_length=1):
        return sorted(
            [
                id
                for id, tracklet in self._tracklets.items()
                if sequence_id == self._tracklets_meta[id]["sequence_id"]
                and len(tracklet) >= min_length
                # self.ds.tgt_ann(tracklet[0])["sequence_id"]
            ],
            key=lambda id: self._tracklets_meta[id]["initial_frame_id"],
            # get_frame_id(self.ds.img_ann(self._get(id)[0])["file_name"]),
        )

    def get_tracklet(self, tracklet_id):
        return self._get(tracklet_id)

    def get_tracklet_sequence_id(self, tracklet_id):
        return self._tracklets_meta[tracklet_id]["sequence_id"]

    def _build_tracklets(self):
        for i in tqdm(range(len(self.ds)), disable=not self.pbar):
            self._add(self.ds.id(i))

    def _get(self, id) -> Tracklet:
        return sorted(
            self._tracklets[id], key=lambda id: fast_get_frame_id(id, self.ds)
        )

    def _add(self, item_id):
        candidates = self._get_candidate_tracklet_ids(item_id)

        if len(candidates) == 0:
            tracklet_id = self._add_new_tracklet(item_id)
        if len(candidates) == 1:
            tracklet_id = self._extend_tracklet(list(candidates)[0], item_id)
        if len(candidates) > 1:
            tracklet_id = self._merge_tracklets(candidates)
            tracklet_id = self._extend_tracklet(tracklet_id, item_id)

        return tracklet_id

    def _new_id(self):
        self._tracklet_counter += 1
        return self._tracklet_counter

    def _create_new_tracklet(self, item_id):
        return Tracklet([item_id])

    def _add_new_tracklet(self, item_id):
        id = self._new_id()
        self._tracklet_ids.append(id)
        self._tracklets[id] = self._create_new_tracklet(item_id)
        self._update_meta(id, item_id)
        return id

    def _extend_tracklet(self, tracklet_id, item_id):
        curr = self._get(tracklet_id)
        self._tracklets[tracklet_id] = curr + self._create_new_tracklet(item_id)
        self._update_meta(tracklet_id, item_id)
        return tracklet_id

    def _merge_tracklets(self, tracklet_ids):
        merge_id = tracklet_ids[0]

        for tracklet_id in tracklet_ids[1:]:
            for item_id in self._get(tracklet_id):
                merged_id = self._extend_tracklet(merge_id, item_id)
            del self._tracklets[tracklet_id]
            del self._tracklets_meta[tracklet_id]
            self._tracklet_ids.remove(tracklet_id)

        self._update_meta(merged_id, self._get(merged_id)[0])

        return merged_id

    def _update_meta(self, tracklet_id, item_id):
        meta = self._tracklets_meta.get(tracklet_id, {})

        prev_sequence_id = meta.get("sequence_id", None)
        new_sequence_id = self.ds.tgt_ann(item_id)["sequence_id"]

        if prev_sequence_id is not None and prev_sequence_id != new_sequence_id:
            raise ValueError(
                f"Tracklet {tracklet_id} cannot belong to sequences {prev_sequence_id} and {new_sequence_id}"
            )

        initial_frame_id = self._get(tracklet_id)[0]

        meta = {
            "sequence_id": new_sequence_id,
            "initial_frame_id": initial_frame_id,
        }

        self._tracklets_meta[tracklet_id] = meta

    def _get_candidate_tracklet_ids(self, item_id):
        sequence_id = lambda id: self.ds.tgt_ann(id)["sequence_id"]
        frame_id = lambda id: fast_get_frame_id(id, self.ds)
        box_iou = lambda a_id, b_id: iou(
            xywh2xyxy(self.ds.tgt_ann(a_id)["bbox"]),
            xywh2xyxy(self.ds.tgt_ann(b_id)["bbox"]),
        )
        return list(
            set(
                [
                    id
                    for id, tracklet in self._tracklets.items()
                    for other_id in tracklet
                    if abs(frame_id(item_id) - frame_id(other_id))
                    <= self.separation_thresh
                    and box_iou(item_id, other_id) >= self.iou_thresh
                    and sequence_id(item_id) == sequence_id(other_id)
                ]
            )
        )


@lru_cache(maxsize=350000)
# The lru_cache drastically improves performance. On a dataset of ~200k samples,
# qualitative experiments show and ETA of 2h30m (growing) without lru_cache,
# while <1h with it. Note: 350000 should accomodate the biggest dataset (342109
# is the theoretical maximum number of positive samples in realcolon).
def fast_get_frame_id(id, ds):
    return get_frame_id(ds.img_ann(id)["file_name"])


def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def iou(a_bbox, b_bbox):
    a_x1, a_y1, a_x2, a_y2 = a_bbox
    b_x1, b_y1, b_x2, b_y2 = b_bbox

    # get the overlap rectangle
    x1 = max(a_x1, b_x1)
    y1 = max(a_y1, b_y1)
    x2 = min(a_x2, b_x2)
    y2 = min(a_y2, b_y2)

    # check if there is an overlap
    if x2 < x1 or y2 < y1:
        return 0

    # if there is an overlap, calculate the area
    intersection_area = (x2 - x1) * (y2 - y1)

    # calculate the area of both boxes
    a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
    b_area = (b_x2 - b_x1) * (b_y2 - b_y1)

    # calculate the union area
    union_area = a_area + b_area - intersection_area

    return intersection_area / union_area
