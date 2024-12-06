import numpy as np
import open3d as o3d


def o3d_check_lines_collision(
    rcs: o3d.t.geometry.RaycastingScene,
    lines_start: np.ndarray,
    lines_end: np.ndarray,
) -> np.ndarray:
    """
    Uses open3d.t.geometry.RaycastingScene to check for collisions between lines and a mesh.

    Args:
        rcs: open3d.t.geometry.RaycastingScene of the mesh
        lines_start: (N,3) array of line start points
        lines_end: (N,3) array of line end points

    Returns:
        (N,) boolean array of whether each line collides with the mesh
    """
    edge_vectors = lines_end - lines_start
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)

    # IMPORTANT: normalize ray direction vector
    ray_directions = edge_vectors / edge_lengths[:, np.newaxis]

    rays = np.concatenate([lines_start, ray_directions], axis=1)
    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    res = rcs.cast_rays(rays)

    collision_mask = np.asarray(res["t_hit"]) < edge_lengths

    return collision_mask
