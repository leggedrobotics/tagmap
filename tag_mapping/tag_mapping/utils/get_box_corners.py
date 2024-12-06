import numpy as np

from open3d.geometry import AxisAlignedBoundingBox, OrientedBoundingBox
from tag_mapping.datasets.matterport import (
    MatterportObjectBoundingBox,
    MatterportRegionBoundingBox,
)


def get_box_corners(box) -> np.ndarray:
    """
    Helper function that takes in a box of a supported type and
    outputs its corners as an array of shape (8, 3) in the following order:

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)

    Args:
        box: a box of a supported type

    Returns:
        corners: (8, 3) array of corners
    """
    if type(box) == AxisAlignedBoundingBox:
        min_bound = box.get_min_bound()
        max_bound = box.get_max_bound()
        corners = np.array(
            [
                min_bound,
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                max_bound,
                [min_bound[0], max_bound[1], max_bound[2]],
            ]
        )

    elif type(box) == OrientedBoundingBox:
        raise NotImplementedError

    elif type(box) == MatterportObjectBoundingBox:
        corners = box.corners()

    elif type(box) == MatterportRegionBoundingBox:
        corners = box.corners()

    else:
        raise ValueError(f"Unsupported box type {type(box)}")

    return corners
