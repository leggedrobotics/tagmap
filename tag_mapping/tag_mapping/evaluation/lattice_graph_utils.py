import numpy as np
import open3d as o3d

from tag_mapping.utils import nearest_points_in_box, o3d_check_lines_collision
from .lattice_navigation_graph import LatticeNavigationGraph


def assign_label_box_lattice_graph_nodes(
    lattice_graph: LatticeNavigationGraph,
    rcs: o3d.t.geometry.RaycastingScene,
    box_corners: np.ndarray,
    enable_inflation: bool = True,
    ogn_dist_threshold: float = 1.0,
):
    """
    Assigns to a labeled bounding box nodes of the lattice graph.
    The assignment includes nodes of the following types:
        - Nodes within the labeled bounding box
        - Nodes who's shortest straight line path to the labeled bounding box
            is within the object goal nav distance threshold and collision free

    The object goal nav distance threshold is nominally defined as 1m following
    the Habitat challenge evaluation criteria:
        https://aihabitat.org/challenge/2023/

    Args:
        lattice_graph: LatticeNavigationGraph
        rcs: Open3d RaycastingScene
        box_corners: (8,3) array of labeled bounding box corners
        enable_inflation: If True, inflate the box to find additional points within
            the object goal nav distance threshold
        ogn_dist_threshold: Object goal nav distance threshold.
            Only used if enable_inflation is True.

    Returns:
        assigned_node_inds: List of node indices assigned to the labeled bounding box.
            Note that this list could be empty if no nodes are assigned
    """
    nodes_xyz = lattice_graph.nodes_xyz

    # Construct the convex hull (i.e. minimum oriented bounding box) from the box_corners
    o3d_box_corners = o3d.geometry.PointCloud()
    o3d_box_corners.points = o3d.utility.Vector3dVector(box_corners)
    obb = o3d_box_corners.get_minimal_oriented_bounding_box()

    # Get the indicies of the nodes within the initial bounding box
    in_box_inds = obb.get_point_indices_within_bounding_box(
        o3d.utility.Vector3dVector(nodes_xyz)
    )

    # Inflate the box to find additional points within the object goal nav distance
    if enable_inflation:
        inflated_obb_extent = obb.extent.copy()
        inflated_obb_extent += 2 * ogn_dist_threshold

        inflated_obb = o3d.geometry.OrientedBoundingBox()
        inflated_obb.center = obb.center.copy()
        inflated_obb.R = obb.R.copy()
        inflated_obb.extent = inflated_obb_extent

        in_inflated_box_inds = inflated_obb.get_point_indices_within_bounding_box(
            o3d.utility.Vector3dVector(nodes_xyz)
        )

        # Consider now only the nodes in the inflated box but NOT in the original box
        near_box_inds = [ind for ind in in_inflated_box_inds if ind not in in_box_inds]

        if len(near_box_inds) > 0:
            near_box_nodes = nodes_xyz[near_box_inds].reshape(-1, 3)
            proj_box_nodes = nearest_points_in_box(
                box_corners, obb.center, near_box_nodes
            )

            near_box_dists = np.linalg.norm(near_box_nodes - proj_box_nodes, axis=1)
            within_ogn_dist = near_box_dists <= ogn_dist_threshold

            collision_mask = o3d_check_lines_collision(
                rcs, near_box_nodes, proj_box_nodes
            )

            valid_mask = np.logical_and(~collision_mask, within_ogn_dist)
            near_box_inds = np.array(near_box_inds)[valid_mask].tolist()

        assigned_node_inds = in_box_inds + near_box_inds
    else:
        assigned_node_inds = in_box_inds

    return assigned_node_inds


def assign_proposal_box_lattice_graph_nodes(
    lattice_graph: LatticeNavigationGraph,
    rcs: o3d.t.geometry.RaycastingScene,
    box_corners: np.ndarray,
):
    """
    Assigns to a proposed bounding box nodes of the lattice graph.

    First we check if the box already contains nodes, if it does then
    we return those nodes. Otherwise we inflate the box and find nodes
    nearby to the box.

    We only assign nearby nodes which are collision free to their projected
    point in the box and who's projected point is within that node's voxel.

    Args:
        lattice_graph: LatticeNavigationGraph
        rcs: Open3d RaycastingScene
        box_corners: (8,3) array of labeled bounding box corners

    Returns:
        assigned_node_inds: List of node indices assigned to the labeled bounding box.
            Note that this list could be empty if no nodes are assigned
    """
    nodes_xyz = lattice_graph.nodes_xyz
    lattice_grid_res = lattice_graph.grid_resolution

    # Construct the convex hull (i.e. minimum oriented bounding box) from the box_corners
    o3d_box_corners = o3d.geometry.PointCloud()
    o3d_box_corners.points = o3d.utility.Vector3dVector(box_corners)
    obb = o3d_box_corners.get_minimal_oriented_bounding_box()

    # Get the indicies of the nodes within the initial bounding box
    in_box_inds = obb.get_point_indices_within_bounding_box(
        o3d.utility.Vector3dVector(nodes_xyz)
    )

    # End if the box already contains nodes
    if len(in_box_inds) > 0:
        return in_box_inds

    # Find additional nodes by inflating the bounding box
    MAGIC_EXTENT_SCALING_CONSTANT = 2 * 1.414  # NOTE: 1.414 ~ sqrt(2)

    inflated_obb_extent = obb.extent.copy()
    inflated_obb_extent = np.maximum(
        inflated_obb_extent, MAGIC_EXTENT_SCALING_CONSTANT * lattice_grid_res
    )

    inflated_obb = o3d.geometry.OrientedBoundingBox()
    inflated_obb.center = obb.center.copy()
    inflated_obb.R = obb.R.copy()
    inflated_obb.extent = inflated_obb_extent

    near_box_inds = inflated_obb.get_point_indices_within_bounding_box(
        o3d.utility.Vector3dVector(nodes_xyz)
    )

    # Assign no nodes if inflating does not find any nodes
    if len(near_box_inds) == 0:
        return []

    near_box_nodes = nodes_xyz[near_box_inds].reshape(-1, 3)
    proj_box_nodes = nearest_points_in_box(box_corners, obb.center, near_box_nodes)

    # Check if the projected points are within the node's voxel using infinity norm
    near_box_dists = np.linalg.norm(near_box_nodes - proj_box_nodes, axis=1, ord=np.inf)
    in_node_voxel = near_box_dists <= (lattice_grid_res / 2)

    collision_mask = o3d_check_lines_collision(rcs, near_box_nodes, proj_box_nodes)

    valid_mask = np.logical_and(~collision_mask, in_node_voxel)
    near_box_inds = np.array(near_box_inds)[valid_mask].tolist()

    return near_box_inds
