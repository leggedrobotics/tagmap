from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class MatterportRegionBoundingBox:
    label: str
    min_bound: np.array
    max_bound: np.array

    def corners(self):
        """
        Corners ordered following convention defined in Pytorch3D
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py
        """
        return np.array(
            [
                self.min_bound,
                [self.max_bound[0], self.min_bound[1], self.min_bound[2]],
                [self.max_bound[0], self.max_bound[1], self.min_bound[2]],
                [self.min_bound[0], self.max_bound[1], self.min_bound[2]],
                [self.min_bound[0], self.min_bound[1], self.max_bound[2]],
                [self.max_bound[0], self.min_bound[1], self.max_bound[2]],
                self.max_bound,
                [self.min_bound[0], self.max_bound[1], self.max_bound[2]],
            ]
        )

    def o3d_lineset(self, color=(0, 1, 0)):
        vertices = self.corners().astype(np.float64)

        lines = np.array(
            [
                [0, 1],
                [0, 3],
                [1, 2],
                [2, 3],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
                [4, 5],
                [4, 7],
                [5, 6],
                [6, 7],
            ]
        ).astype(np.int32)

        colors = np.tile(color, (12, 1)).astype(np.float64)

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(vertices)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector(colors)

        return lineset
