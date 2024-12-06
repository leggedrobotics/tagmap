from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class MatterportObjectBoundingBox:
    category_index: int
    center: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    r: np.ndarray

    def corners(self):
        """
        Corners ordered following convention defined in Pytorch3D
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py
        """
        a1 = self.a1 / np.linalg.norm(self.a1)
        a2 = self.a2 / np.linalg.norm(self.a2)
        r1, r2, r3 = self.r
        a3 = np.cross(self.a1, self.a2)
        return np.array(
            [
                self.center - r1 * a1 - r2 * a2 - r3 * a3,
                self.center + r1 * a1 - r2 * a2 - r3 * a3,
                self.center + r1 * a1 + r2 * a2 - r3 * a3,
                self.center - r1 * a1 + r2 * a2 - r3 * a3,

                self.center - r1 * a1 - r2 * a2 + r3 * a3,
                self.center + r1 * a1 - r2 * a2 + r3 * a3,
                self.center + r1 * a1 + r2 * a2 + r3 * a3,
                self.center - r1 * a1 + r2 * a2 + r3 * a3,
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
