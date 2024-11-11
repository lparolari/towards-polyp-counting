import csv
import os
import pathlib
import re
from collections import namedtuple
from typing import Tuple
from xml.etree import ElementTree


def parsevocfile(annotation_file):
    """Parse an annotation file in voc format

        Example VOC notation:
            <annotation>
                </version_fmt>1.0<version_fmt>
                <folder>002-001_frames</folder>
                <filename>002-001_18185.jpg</filename>
                <source>
                    <database>cosmoimd</database>
                    <release>v1.0_20230228</release>
                </source>
                <size>
                    <width>1240</width>
                    <height>1080</height>
                    <depth>3</depth>
                </size>
                <object>
                    <name>lesion</name>
                    <unique_id>videoname_lesionid</unique_id>
                    <box_id>1</box_id>  <- id of the box within the image
                    <bndbox>
                        <xmin>540</xmin>
                        <xmax>1196</xmax>
                        <ymin>852</ymin>
                        <ymax>1070</ymax>
                    </bndbox>
                </object>
            </annotation>""

    Args:
        ann_filename (string) : Full path to the file to parse

    Returns:
        dict: The list of boxes for each class and the image shape
    """

    if not os.path.exists(annotation_file):
        raise Exception("Cannot find bounding box file %s" % (annotation_file))
    try:
        tree = ElementTree.parse(annotation_file)
    except Exception as e:
        print(e)
        raise Exception("Failed to open annotation file %s" % annotation_file)

    # Read all the boxes
    img = {}
    cboxes = []
    for elem in tree.iter():
        # Get the image full path from the image name and folder, not from the annotation tag
        if "filename" in elem.tag:
            filename = elem.text
        if "width" in elem.tag:
            img["width"] = int(elem.text)
        if "height" in elem.tag:
            img["height"] = int(elem.text)
        if "depth" in elem.tag:
            img["depth"] = int(elem.text)
        if "object" in elem.tag or "part" in elem.tag:
            obj = {}

            # create empty dict where store properties
            for attr in list(elem):
                if "name" in attr.tag:
                    obj["name"] = attr.text
                if "unique_id" in attr.tag:
                    obj["unique_id"] = attr.text

                if "bndbox" in attr.tag:
                    for dim in list(attr):
                        if "xmin" in dim.tag:
                            l = int(round(float(dim.text)))
                        if "ymin" in dim.tag:
                            t = int(round(float(dim.text)))
                        if "xmax" in dim.tag:
                            r = int(round(float(dim.text)))
                        if "ymax" in dim.tag:
                            b = int(round(float(dim.text)))

                    obj["box_ltrb"] = [l, t, r, b]
            cboxes.append(obj)
    img_shape = (img["height"], img["width"], img["depth"])
    return {"boxes": cboxes, "img_shape": img_shape, "img_name": filename}


def get_frame_id(frame_path) -> int:
    """
    Return the frame number identifier.
    """
    pattern = r"\d+-\d+_(\d+)(?:\.\d+)?"

    match = re.search(pattern, frame_path)

    if not match:
        return None

    return int(match.group(1))


def get_frame_name(frame_name: str) -> str:
    """
    Return the frame unique name.
    """
    pattern = r"(\d+-\d+_\d+(\.\d+)?)"

    match = re.search(pattern, frame_name)

    if not match:
        return None

    return match.group(1)


def load_files_list(files_dir):
    """
    Load the list of files in a directory and sort them by frame id

    Args:
        files_dir (Path): Path to the directory containing the files

    Returns:
        list: List of files sorted by frame id
    """
    files_dir = pathlib.Path(files_dir)
    files_list = [files_dir / f for f in os.listdir(files_dir)]
    files_list = sorted(files_list, key=lambda x: get_frame_id(x.name))
    return files_list


VideoInfo = namedtuple(
    "VideoInfo",
    [
        "unique_video_name",
        "age",
        "sex",
        "endoscope_brand",
        "fps",
        "num_frames",
        "num_lesions",
        "bbps",
    ],
)

DatasetInfo = list[VideoInfo]


def get_info(info_csv) -> DatasetInfo:
    # example row: "004-006",79,"female","Olympus",29.97,40986,1,8.5
    with open(info_csv) as f:
        reader = csv.reader(f, quotechar='"')
        next(reader)  # skip header
        return [
            VideoInfo(
                unique_video_name=row[0],
                age=int(row[1]),
                sex=row[2],
                endoscope_brand=row[3],
                fps=float(row[4]),
                num_frames=int(row[5]),
                num_lesions=int(row[6]),
                bbps=float(row[7]),
            )
            for row in reader
        ]


def filter_videos(info: DatasetInfo, video_name_regex: str) -> DatasetInfo:
    return [
        video for video in info if re.match(video_name_regex, video.unique_video_name)
    ]


def real_colon_split(
    instances, train_videos, val_videos, test_videos
) -> Tuple[dict, dict, dict]:
    """
    Partition given instances into train, validation and test sets based on
    video ids. Matching is done on the image file name, which is expected to be
    in the format "<video_id>_<frame_id>.<ext>".
    """
    imgid2annidx = {}
    for i, ann in enumerate(instances["annotations"]):
        if ann["image_id"] not in imgid2annidx:
            imgid2annidx[ann["image_id"]] = []
        imgid2annidx[ann["image_id"]].append(i)

    split_out = []

    for split_videos in [train_videos, val_videos, test_videos]:
        split_instances = {
            **instances,
            "images": [],
            "annotations": [],
        }

        for image in instances["images"]:
            video_id = image["file_name"].split("_")[0]
            if video_id in split_videos:
                anns_by_img = imgid2annidx.get(image["id"], [])
                split_instances["images"].append(image)
                split_instances["annotations"].extend(
                    [instances["annotations"][i] for i in anns_by_img]
                )

        split_out.append(split_instances)

    return tuple(split_out)
