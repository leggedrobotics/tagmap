from typing import Iterable, Optional, Tuple
from .viewpoint import Viewpoint

import numpy as np


def voxel_voting(
    points: np.ndarray,
    voxel_size: float,
    point_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voxel voting based localization given a set of points.

    Args:
        points: (N,3) array of points
        voxel_size: Size of the voxels
        point_weights: (N,) array of weights for each point. If None, all points
            are weighted equally.

    Returns:
        A point cloud of the voxel centers (N,3) and the votes for each voxel (N,)
    """
    voxel_coords = np.floor(points / voxel_size)
    keys = voxel_coords.astype(np.int32)
    _, inds, inverse_inds, counts = np.unique(
        keys, axis=0, return_index=True, return_inverse=True, return_counts=True
    )

    if point_weights is None:
        votes = counts
    else:
        votes = np.zeros(len(inds))
        for i in range(len(inds)):
            votes[i] = np.sum(point_weights[inverse_inds == i])

    voxel_centers = (voxel_coords[inds] + 0.5) * voxel_size

    return voxel_centers, votes


def grid_voxel_voting(
    viewpoints: Iterable[Viewpoint],
    voxel_size: float,
    viewpoint_weight: Optional[Iterable[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Voxel voting based localization given a set of viewpoints.
    1. For each viewpoint, generate a set of grid points inside the viewpoint
        where the grid is aligned with the world frame
    2. Merge grid points into a single point cloud
    3. Voxelize the point cloud and count the number of points in each voxel
        where the count can be weighted by viewpoint_weight

    Args:
        viewpoints: Iterable of viewpoints
        voxel_size: Size of the voxels
        viewpoint_weight: Iterable of weights for each viewpoint. If None, all
            viewpoints are weighted equally.

    Returns:
        A point cloud of the voxel centers (N,3) and the votes for each voxel (N,)
    """
    if viewpoint_weight is not None:
        assert len(viewpoint_weight) == len(viewpoints)

    grid_points = []
    grid_point_weights = []
    for i, vp in enumerate(viewpoints):
        aabb_max_bound = vp.aabb.get_max_bound()
        aabb_min_bound = vp.aabb.get_min_bound()

        range_N = np.ceil((aabb_max_bound - aabb_min_bound) / voxel_size)

        xx, yy, zz = np.meshgrid(
            voxel_size * np.arange(range_N[0]) + aabb_min_bound[0],
            voxel_size * np.arange(range_N[1]) + aabb_min_bound[1],
            voxel_size * np.arange(range_N[2]) + aabb_min_bound[2],
        )

        vp_grid_points = np.concatenate(
            [c.reshape(-1, 1) for c in [xx, yy, zz]], axis=1
        )

        # get only the points within the viewpoint
        inside = vp.within_viewpoint(vp_grid_points)
        vp_grid_points = vp_grid_points[inside]

        grid_points.append(vp_grid_points)

        if viewpoint_weight is not None:
            grid_point_weights.append(
                viewpoint_weight[i] * np.ones(vp_grid_points.shape[0])
            )

    if len(grid_points) == 0:
        return np.zeros((0, 3)), np.zeros((0,))

    grid_points = np.concatenate(grid_points, axis=0)
    grid_point_weights = (
        None if viewpoint_weight is None else np.concatenate(grid_point_weights, axis=0)
    )

    return voxel_voting(grid_points, voxel_size, grid_point_weights)
