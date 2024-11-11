from polypsense.dataset.realcolon import get_frame_id
from polypsense.mve.trackleter import Trackleter


def get_polyps(annotation_list):
    """
    Given a list of annotations (e.g. output of parsevocfile), return a list
    where each item is a polyp sample and has the information on for a single
    polyp annotation.

    The returned polyp's annotation has the following structure:
    {
        "frame_id": 1,
        "frame_name": "001-001_00001.jpg",
        "frame_shape": (h, w, 3),
        "bbox": [100, 100, 200, 200],
        "sequence": "001-001_1"
    }
    """
    polyps = []

    for i, info in enumerate(annotation_list):
        frame_id = get_frame_id(info["img_name"])

        assert i == frame_id, f"Found frame_id {frame_id} that differs from index {i}"

        for box_info in info["boxes"]:

            polyps.append(
                {
                    "frame_id": frame_id,
                    "frame_name": info["img_name"],
                    "frame_shape": info["img_shape"],
                    "bbox": box_info["box_ltrb"],  # xyxy
                    "sequence": box_info["unique_id"],
                }
            )

    return polyps


class Polyps2Dataset:
    def __init__(self, polyps):
        self.polyps = polyps
        self.ids = list(range(len(polyps)))

    def __getitem__(self, idx):
        return self.polyps[self.id(idx)]

    def __len__(self):
        return len(self.polyps)

    def id(self, idx):
        return self.ids[idx]

    def tgt_ann(self, id):
        def xyxy2xywh(bbox):
            return [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

        return {
            "bbox": xyxy2xywh(self.polyps[id]["bbox"]),
            "sequence_id": self.polyps[id]["sequence"],
        }

    def img_ann(self, id):
        return {
            "id": id,
            "file_name": self.polyps[id]["frame_name"],
            # f"001-001_{self.polyps[id]['frame_id']}.jpg",
        }

    def img(self, id):
        raise NotImplementedError


def polyps2ds(polyps):
    return Polyps2Dataset(polyps)


class FragmentationRateDataset:
    """
    Exposes the interface for fragmentation rate routines.
    """

    def __init__(self, polyps):
        self.ds = polyps2ds(polyps)
        self.trackleter = Trackleter(self.ds)

    def __len__(self):
        return len(self.trackleter)

    def __getitem__(self, idx):
        return self.trackleter[idx]

    def polyp(self, polyp_id):
        tgt_ann = self.ds.tgt_ann(polyp_id)
        img_ann = self.ds.img_ann(polyp_id)

        return {
            "frame_id": get_frame_id(img_ann["file_name"]),
            "frame_name": img_ann["file_name"],
            "bbox": tgt_ann["bbox"],
            "sequence": tgt_ann["sequence_id"],
        }

    def gap(self, idx1, idx2):
        """
        Return the gap between two tracklets. The gap is defined as the number
        of frames between the last frame of the first tracklet and the first
        frame of the second tracklet.

        If the gap is negative, it means that the tracklets overlap and the
        distance is more negative the more the two tracklets overlap.
        """
        if idx1 == idx2:
            return 0

        t1 = self[idx1]
        t2 = self[idx2]

        t1_right = self.polyp(t1[-1])["frame_id"]
        t2_left = self.polyp(t2[0])["frame_id"]

        return t2_left - t1_right
