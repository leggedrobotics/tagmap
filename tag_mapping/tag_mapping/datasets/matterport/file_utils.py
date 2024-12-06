import numpy as np
from PIL import Image
from typing import Tuple, List


def read_matterport_image_file(image_filepath) -> Image.Image:
    image = Image.open(image_filepath)
    return image


def read_matterport_depth_file(depth_filepath) -> Tuple[np.ndarray, Image.Image]:
    depth_image = Image.open(depth_filepath)
    SCALE_FACTOR = 4000  # https://github.com/vsislab/Matterport3D-Layout/issues/4
    depth = np.asarray(depth_image) / SCALE_FACTOR
    return depth, depth_image


def read_matterport_pose_file(pose_filepath) -> np.ndarray:
    # NOTE: returns the pose from camera frame to the world frame
    with open(pose_filepath, "r") as file:
        lines = file.readlines()
    T_cam_to_world = np.array(
        [list(map(float, line.split(" ")[:-1])) for line in lines]
    )
    return T_cam_to_world


def read_matterport_intrinsics_file(intrinsics_filepath):
    with open(intrinsics_filepath, "r") as file:
        lines = file.readlines()
    line = lines[0]
    intrinsics = line.split(" ")

    width = int(intrinsics[0])
    height = int(intrinsics[1])
    fx = float(intrinsics[2])
    fy = float(intrinsics[3])
    cx = float(intrinsics[4])
    cy = float(intrinsics[5])
    d = [float(i) for i in intrinsics[6:]]

    return width, height, fx, fy, cx, cy, d


class MatterportFilenameBridge:
    def __init__(self, frame_identifiers):
        self._frame_identifiers = frame_identifiers

    @classmethod
    def from_image_filename(cls, image_filename):
        frame_identifiers = image_filename.split(".")[0].split("_")
        frame_identifiers[1] = frame_identifiers[1][1:]
        return cls(frame_identifiers)

    @classmethod
    def from_pose_filename(cls, pose_filename):
        frame_identifiers = pose_filename.split(".")[0].split("_")
        frame_identifiers.remove("pose")
        return cls(frame_identifiers)

    @property
    def image_filename(self):
        return (
            self._frame_identifiers[0]
            + "_i"
            + self._frame_identifiers[1]
            + "_"
            + self._frame_identifiers[2]
            + ".jpg"
        )

    @property
    def depth_filename(self):
        return (
            self._frame_identifiers[0]
            + "_d"
            + self._frame_identifiers[1]
            + "_"
            + self._frame_identifiers[2]
            + ".png"
        )

    @property
    def pose_filename(self):
        return (
            self._frame_identifiers[0]
            + "_pose_"
            + self._frame_identifiers[1]
            + "_"
            + self._frame_identifiers[2]
            + ".txt"
        )


import re
from .matterport_object_bounding_box import MatterportObjectBoundingBox

from .category_mapping import load_category_index_mapping

MATTERPORT_CATEGORY_INDEX_MAPPING = load_category_index_mapping()


def read_matterport_object_bounding_boxes(
    house_filepath,
    category_taxonomies=("category", "mpcat40"),
):
    """
    Reads the object bounding box labels in a .house file.

    Args:
        house_filepath: A string that specifies the path to the .house file.
        category_taxonomies: A tuple of strings that specifies the taxonomies to use for
            categorizing the objects.

    Returns:
        A dictionary that maps each taxonomy to a dictionary mapping that taxonomy's labels
            to a list of the corresponding bounding boxes.
    """
    for t in category_taxonomies:
        assert t in MATTERPORT_CATEGORY_INDEX_MAPPING[0], "Invalid taxonomy {}".format(
            t
        )

    with open(house_filepath) as house_file:
        lines = house_file.readlines()

    boxes = []
    for line in lines:
        if line[0] == "O":
            data = re.split(r" +", line)

            boxes.append(
                MatterportObjectBoundingBox(
                    category_index=int(data[3]),
                    center=np.array(data[4:7], dtype=float),
                    a1=np.array(data[7:10], dtype=float),
                    a2=np.array(data[10:13], dtype=float),
                    r=np.array(data[13:16], dtype=float),
                )
            )

    out = {t: {} for t in category_taxonomies}
    for box in boxes:
        for t in category_taxonomies:
            try:
                t_label = MATTERPORT_CATEGORY_INDEX_MAPPING[box.category_index][t]
                if t_label not in out[t]:
                    out[t][t_label] = [box]
                else:
                    out[t][t_label].append(box)
            except KeyError:
                print(
                    "[warning]: bounding box with invalid category_index {}".format(
                        box.category_index
                    )
                )

    return out


from plyfile import PlyData


def read_matterport_labeled_points(
    mesh_ply_filepath,
    category_taxonomies=("category", "mpcat40"),
):
    """
    Gets labeled points from the .ply house segmentation file.
    Since the faces of the mesh are labeled, we compute the points as the mean of the vertices of the faces.

    Args:
        house_filepath: A string that specifies the path to the .house file.
        category_taxonomies: A tuple of strings that specifies the taxonomies to use for
            categorizing the objects.

    Returns:
        First returns a dictionary that maps each taxonomy to a dictionary mapping that taxonomy's labels
            to a list of the corresponding points. Second, returns the points as a numpy array.
    """
    for t in category_taxonomies:
        assert t in MATTERPORT_CATEGORY_INDEX_MAPPING[0], "Invalid taxonomy {}".format(
            t
        )

    plydata = PlyData.read(mesh_ply_filepath)
    vertex_xyz = np.zeros((len(plydata["vertex"]), 3))
    vertex_xyz[:, 0] = plydata["vertex"]["x"]
    vertex_xyz[:, 1] = plydata["vertex"]["y"]
    vertex_xyz[:, 2] = plydata["vertex"]["z"]

    face_vertex_inds = np.vstack(  # NOTE: this computation is slow
        plydata["face"]["vertex_indices"]
    )
    face_center_xyz = np.mean(vertex_xyz[face_vertex_inds], axis=1)
    face_category_ids = plydata["face"]["category_id"]

    unique_category_ids = np.unique(face_category_ids)

    out = {t: {} for t in category_taxonomies}
    for category_id in unique_category_ids:
        for t in category_taxonomies:
            try:
                # NOTE: use category_id - 1 because for some reason the ply file
                #      has category ids that are 1-indexed
                t_label = MATTERPORT_CATEGORY_INDEX_MAPPING[category_id - 1][t]
                face_inds = np.where(face_category_ids == category_id)[0]

                if t_label not in out[t]:
                    out[t][t_label] = face_inds
                else:
                    out[t][t_label] = np.concatenate((out[t][t_label], face_inds))
            except KeyError:
                print("[warning]: face with invalid category_id {}".format(category_id))

    return out, face_center_xyz


# https://github.com/niessner/Matterport/blob/master/data_organization.md
# fmt: off
MATTERPORT_REGION_NAME_MAPPING = {
    "a": "bathroom",  # (should have a toilet and a sink)
    "b": "bedroom",
    "c": "closet",
    "d": "dining room",  # (includes “breakfast rooms” other rooms people mainly eat in)
    "e": "entryway/foyer/lobby",  # (should be the front door, not any door)

    # "f": "familyroom",  # (should be a room that a family hangs out in, not any area with couches)
    "f": "living room",
    
    "g": "garage",
    "h": "hallway",
    "i": "library",  # (should be room like a library at a university, not an individual study)
    "j": "laundryroom/mudroom",  # (place where people do laundry, etc.)
    "k": "kitchen",
    "l": "living room",  # (should be the main “showcase” living room in a house, not any area with couches)
    "m": "meetingroom/conferenceroom",

    # "n": "lounge",  # (any area where people relax in comfy chairs/couches that is not the family room or living room
    "n": "living room",

    "o": "office",  # (usually for an individual, or a small set of people)
    "p": "porch/terrace/deck/driveway",  # (must be outdoors on ground level)
    "r": "rec/game",  # (should have recreational objects, like pool table, etc.)
    "s": "stairs",

    # "t": "toilet",  # (should be a small room with ONLY a toilet)
    "t": "bathroom",

    "u": "utilityroom/toolroom",
    "v": "tv",  # (must have theater-style seating)
    "w": "workout/gym/exercise",
    "x": "outdoor",  # areas containing grass, plants, bushes, trees, etc.
    "y": "balcony",  # (must be outside and must not be on ground floor)
    "B": "bar",
    "C": "classroom",
    "D": "dining booth",
    "S": "spa/sauna",

    "z": "other room",  # (it is clearly a room, but the function is not clear)
    "Z": "junk",  # (reflections of mirrors, random points floating in space, etc.)
    "-": "no label",
}
# fmt: on

from .matterport_region_bounding_box import MatterportRegionBoundingBox


def read_matterport_region_bounding_boxes(
    house_filepath,
):
    """
    Reads the region bounding boxes in a .house file.

    Args:
        house_filepath: A string that specifies the path to the .house file.

    Returns:
        A dictionary that maps each region label to a list of the corresponding bounding boxes.
    """
    with open(house_filepath) as house_file:
        lines = house_file.readlines()

    out = {}
    for line in lines:
        if line[0] == "R":
            data = re.split(r" +", line)

            min_bound = np.array(data[9:12], dtype=float)
            max_bound = np.array(data[12:15], dtype=float)
            if np.any(min_bound == max_bound):
                PAD = 1.0
                print(
                    f"warning: region label box with zero volumn, padding each bound by {PAD}"
                )
                min_bound = min_bound - PAD
                max_bound = max_bound + PAD

            box = MatterportRegionBoundingBox(
                label=MATTERPORT_REGION_NAME_MAPPING[data[5]],
                min_bound=min_bound,
                max_bound=max_bound,
            )

            if box.label not in out:
                out[box.label] = [box]
            else:
                out[box.label].append(box)
    return out
