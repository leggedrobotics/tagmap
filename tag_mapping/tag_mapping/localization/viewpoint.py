from typing import Optional, Tuple, Union

import numpy as np
import open3d as o3d


class Viewpoint:
    """
    A viewpoint where the local frame has Z forward
    following traditional camera frame conventions
    """

    def __init__(
        self,
        extrinsic_matrix: np.ndarray,
        w_fov: float,
        h_fov: float,
        near_dist: Optional[float] = None,
        far_dist: Optional[float] = None,
        extras: Optional[dict] = None,
    ) -> None:
        """
        Args:
            extrinsic_matrix: 4x4 transformation matrix from local to world frame
            w_fov: horizontal field of view in radians
            h_fov: vertical field of view in radians
            near_dist: distance to near plane
            far_dist: distance to far plane
            extras: optional dictionary of extra data
        """
        self._extrinsic_matrix = extrinsic_matrix
        self._w_fov, self._h_fov = (w_fov, h_fov)

        self._near_dist, self._far_dist = (near_dist, far_dist)
        if near_dist != None and far_dist != None:
            assert near_dist < far_dist, "near_dist must be less than far_dist"

        self._extras = extras

    @classmethod
    def from_intrinsics(
        cls,
        extrinsic_matrix: np.ndarray,
        width: int,
        height: int,
        fx: float,
        fy: float,
        near_dist: Optional[float] = None,
        far_dist: Optional[float] = None,
        extras: Optional[dict] = None,
    ) -> "Viewpoint":
        """
        Construct a viewpoint for camera intrinsics parameters
        """
        return cls(
            extrinsic_matrix,
            2 * np.arctan(width / (2 * fx)),
            2 * np.arctan(height / (2 * fy)),
            near_dist,
            far_dist,
            extras,
        )

    def within_viewpoint(self, points: np.ndarray) -> np.ndarray:
        """
        Returns a boolean array indicating whether each point is within the
        viewpoint's frustum

        Args:
            points: Nx3 array of points in the world frame

        Returns:
            (N,) boolean array
        """
        # transform points to the viewpoint's local frame
        points_local = (points - self.origin) @ self.R

        # compute w and h ray angles for the grid points and filter out
        # points which are outside the field of view
        w = np.arctan(points_local[:, 0] / (points_local[:, 2] + 1e-6))
        h = np.arctan(points_local[:, 1] / (points_local[:, 2] + 1e-6))

        inside = np.logical_and(
            np.logical_and(w < (self._w_fov / 2), w > -(self._w_fov / 2)),
            np.logical_and(h < (self._h_fov / 2), h > -(self._h_fov / 2)),
        )

        # check depth bounds
        d = points_local[:, 2]

        if self._near_dist != None:
            inside = np.logical_and(inside, d > self._near_dist)
        else:
            inside = np.logical_and(inside, d > 0)

        if self._far_dist != None:
            inside = np.logical_and(inside, d < self._far_dist)

        return inside

    @property
    def extras(self) -> Union[dict, None]:
        return self._extras

    @property
    def R(self) -> np.ndarray:
        """
        Rotation matrix from the viewpoint's local frame to the world frame
        """
        return self._extrinsic_matrix[:3, :3]

    @property
    def origin(self) -> np.ndarray:
        """
        The origin of the viewpoint's local frame in the world frame
        """
        return self._extrinsic_matrix[:3, 3]

    @property
    def fov(self) -> Tuple[float, float]:
        """
        Returns the horizontal and vertical field of view in radians
        """
        return self._w_fov, self._h_fov

    @property
    def bounding_rays(self) -> np.ndarray:
        """
        Returns the four bounding rays expressed
        in the world frame

        Returns a 4x3 array of unit vectors with each row
        representing a ray in the order of:
            top-left corner
            bottom-left corner
            top-right corner
            bottom-right corner
        """
        if not hasattr(self, "_bounding_rays"):
            dx, dy = (np.tan(self._w_fov / 2), np.tan(self._h_fov / 2))

            rays = np.array(
                [
                    [-dx, -dy, 1.0],  # top-left corner
                    [-dx, dy, 1.0],  # bottom-left corner
                    [dx, -dy, 1.0],  # top-right corner
                    [dx, dy, 1.0],  # bottom-right corner
                ]
            )
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)

            # transform rays to the world frame
            self._bounding_rays = rays @ self._extrinsic_matrix[:3, :3].T

        return self._bounding_rays

    @property
    def frustum_points(self) -> np.ndarray:
        """
        Returns the eight points of the viewpoint's frustum

        Returns a 8x3 array of points in the order of:
            near top-left corner
            near bottom-left corner
            near top-right corner
            near bottom-right corner

            far top-left corner
            far bottom-left corner
            far top-right corner
            far bottom-right corner

        NOTE: if near_dist or far_dist are None, the corresponding far and near
        dists are set to 0 and 1 respectively
        """
        if not hasattr(self, "_frustum_points"):
            ez_world = self.R[:, -1]
            d = ez_world.T @ self.bounding_rays[0]
            near_factor = 0 if self._near_dist == None else self._near_dist / d
            far_factor = 1 if self._far_dist == None else self._far_dist / d

            self._frustum_points = np.array(
                [
                    self.origin + self.bounding_rays[0] * near_factor,
                    self.origin + self.bounding_rays[1] * near_factor,
                    self.origin + self.bounding_rays[2] * near_factor,
                    self.origin + self.bounding_rays[3] * near_factor,
                    self.origin + self.bounding_rays[0] * far_factor,
                    self.origin + self.bounding_rays[1] * far_factor,
                    self.origin + self.bounding_rays[2] * far_factor,
                    self.origin + self.bounding_rays[3] * far_factor,
                ]
            ).astype(np.float64)

        return self._frustum_points

    @property
    def aabb(self) -> o3d.geometry.AxisAlignedBoundingBox:
        """
        Returns the axis-aligned bounding box of the viewpoint's frustum
        """
        if not hasattr(self, "_aabb"):
            self._aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(self.frustum_points)
            )
        return self._aabb

    def o3d_lineset(self, color=(0, 0, 1)) -> o3d.geometry.LineSet:
        """
        Return the an Open3D lineset of the viewpoint for visualization
        """
        points = self.frustum_points

        lines = np.array(
            [
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],
            ]
        ).astype(np.int32)

        if self._near_dist != None:
            near_plane_lines = np.array(
                [
                    [0, 1],
                    [0, 2],
                    [1, 3],
                    [2, 3],
                ]
            ).astype(np.int32)
            lines = np.concatenate([lines, near_plane_lines], axis=0)

        if self._far_dist != None:
            far_plane_lines = np.array(
                [
                    [4, 5],
                    [4, 6],
                    [5, 7],
                    [6, 7],
                ]
            ).astype(np.int32)
            lines = np.concatenate([lines, far_plane_lines], axis=0)

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(points)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector(np.tile(color, (lines.shape[0], 1)))

        return lineset
