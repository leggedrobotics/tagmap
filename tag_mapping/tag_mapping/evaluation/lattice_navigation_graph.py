import time
import os
import numpy as np
import open3d as o3d
import networkx as nx
import pickle

from dataclasses import dataclass
from typing import Tuple, Set, Union, List
from scipy.spatial import KDTree
from tqdm import tqdm


@dataclass(frozen=True)
class LatticeNavigationGraph:
    """
    Lattice navigation graph for a given scene mesh,
    used to evaluate the region reachability metric
    """

    _graph: nx.Graph
    _nodes_xyz: np.ndarray
    _spl_array: np.ndarray
    _grid_resolution: float
    _generation_params: dict

    def shortest_path_length(
        self, node1_ind: int, node2_ind: int
    ) -> Union[float, None]:
        """
        Get the shortest path length between two nodes in the graph.

        Args:
            node1_ind: Index of the first node
            node2_ind: Index of the second node

        Returns:
            The shortest path length between the two nodes, or None if no path exists
        """
        spl = self._spl_array[node1_ind, node2_ind]
        if spl == 0:
            spl = None
        else:
            spl = (spl - 1.0) * self._grid_resolution
        return spl

    def batch_shortest_path_length(
        self, node1_inds: np.ndarray, node2_inds: np.ndarray
    ) -> np.ndarray:
        """
        Get the shortest path lengths for a batch of node pairs.

        NOTE: The values are queried from self._spl_array, which uses 0 to indicate
        if there's no path between two nodes. Therefore, we subtract 1 from the values
        to get the actual shortest path length.

        Args:
            node1_inds: (B,) array of indices of the first nodes
            node2_inds: (B,) array of indices of the second nodes

        Returns:
            (B,) array of shortest path lengths. Values are np.inf if no path exists
        """
        assert node1_inds.shape == node2_inds.shape
        assert node1_inds.ndim == 1

        spl = self._spl_array[node1_inds, node2_inds].copy().astype(np.float32)
        spl[spl == 0] = np.inf
        spl = (spl - 1.0) * self._grid_resolution
        return spl

    def shortest_path(self, node1_ind: int, node2_ind: int) -> Union[List, None]:
        """
        Get the shortest path between two nodes in the graph.

        Args:
            node1_ind: Index of the first node
            node2_ind: Index of the second node

        Returns:
            The shortest path between the two nodes in the form of a list of node indices,
                or None if no path exists
        """
        try:
            sp = nx.shortest_path(self._graph, node1_ind, node2_ind)
        except nx.NetworkXNoPath:
            sp = None
        return sp

    def save(self, path):
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as file:
            obj = pickle.load(file)
        if isinstance(obj, cls):
            return obj
        else:
            raise ValueError(
                "Loaded object is not an instance of LatticeNavigationGraph"
            )

    @property
    def grid_resolution(self):
        return self._grid_resolution

    @property
    def nodes_xyz(self):
        return self._nodes_xyz.copy()

    @property
    def o3d_nodes_pointcloud(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._nodes_xyz.copy())
        return pcd

    @property
    def o3d_edges_lineset(self):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(self._nodes_xyz.copy())
        lines = np.array(self._graph.edges).copy()
        lineset.lines = o3d.utility.Vector2iVector(lines)
        return lineset


def create_lattice_navigation_graph(
    mesh: o3d.geometry.TriangleMesh,
    params: dict,
    print_progress: bool = False,
) -> LatticeNavigationGraph:
    """
    Create a lattice navigation graph for a given mesh.

    Args:
        mesh: o3d.geometry.TriangleMesh
        params: Dictionary of parameters
        print_progress: Whether to print progress

    Returns:
        a LatticeNavigationGraph
    """
    if print_progress:
        print("Starting generate lattice grid")
        start_time = time.time()
    nodes_xyz, edges = _generate_lattice_grid(mesh, **params["lattice_grid_kwargs"])
    if print_progress:
        print(f"Finished in {time.time() - start_time:.2f}s")

    if print_progress:
        print("Starting filtering out nodes and edges outside the mesh")
        start_time = time.time()
    valid_nodes_mask = _find_points_within_mesh(
        nodes_xyz, mesh, **params["lattice_filter_kwargs"]
    )
    old_node_inds = np.where(valid_nodes_mask == True)[0]
    filt_edges = []
    for e in edges:
        if valid_nodes_mask[e[0]] == False or valid_nodes_mask[e[1]] == False:
            continue
        filt_edges.append(
            (
                np.where(old_node_inds == e[0])[0].item(),
                np.where(old_node_inds == e[1])[0].item(),
            )
        )
    filt_edges = np.array(filt_edges).astype(np.int32)
    filt_nodes_xyz = nodes_xyz[valid_nodes_mask]
    if print_progress:
        print(f"Finished in {time.time() - start_time:.2f}s")

    if print_progress:
        print("Starting filtering out edges intersecting mesh")
        start_time = time.time()
    edge_intersects_mesh = _find_mesh_intersecting_edges(
        mesh, filt_nodes_xyz, filt_edges
    )
    free_edges = filt_edges[~edge_intersects_mesh]
    if print_progress:
        print(f"Finished in {time.time() - start_time:.2f}s")

    G = nx.Graph()
    G_node_inds = np.arange(filt_nodes_xyz.shape[0]).astype(np.int32)
    G.add_nodes_from(G_node_inds)
    G.add_edges_from(free_edges.astype(np.int32))

    if print_progress:
        print("Starting filling shortest path lengths array")
    num_nodes = G.number_of_nodes()
    spl_array = np.zeros((num_nodes, num_nodes), dtype=np.uint16)
    progress_bar = tqdm(total=num_nodes)
    for ind1, path_lengths in nx.all_pairs_shortest_path_length(G):
        for ind2, lengths in path_lengths.items():
            # NOTE: Use 0 to indicate if there's no path since int arrays don't support np.inf
            # Store length + 1 to avoid confusion with 0 indicating no path
            spl_array[ind1, ind2] = lengths + 1
        progress_bar.update(1)
    if print_progress:
        print(f"Finished in {time.time() - start_time:.2f}s")

    return LatticeNavigationGraph(
        _graph=G,
        _nodes_xyz=filt_nodes_xyz,
        _spl_array=spl_array,
        _grid_resolution=params["lattice_grid_kwargs"]["grid_res"],
        _generation_params=params,
    )


def _generate_lattice_grid(
    mesh: o3d.geometry.TriangleMesh, grid_res: float = 0.5, outer_pad: float = 0.1
) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:
    """
    Generate an axis-aligned lattice grid on the space covered by the mesh.

    Args:
        mesh: o3d.geometry.TriangleMesh
        grid_res: Resolution/grid edge length of the lattice
        outer_pad: Padding to add to the outer bounds of the mesh

    Returns:
        nodes_xyz: (N,3) array of node xyz coordinates
        edges: Set of edges as tuples of node indices
    """
    min_bound = mesh.get_min_bound() - outer_pad
    max_bound = mesh.get_max_bound() + outer_pad
    num_grid_points = np.ceil((max_bound - min_bound) / grid_res).astype(np.int32)

    xx, yy, zz = np.meshgrid(
        grid_res * np.arange(num_grid_points[0]) + min_bound[0],
        grid_res * np.arange(num_grid_points[1]) + min_bound[1],
        grid_res * np.arange(num_grid_points[2]) + min_bound[2],
        # Use ij indexing such that X changes slowest, Y faster, Z fastest
        indexing="ij",
    )
    nodes_xyz = np.concatenate([c.reshape(-1, 1) for c in [xx, yy, zz]], axis=1)

    def flatten_xyz_inds(x_ind, y_ind, z_ind):
        return (
            z_ind
            + y_ind * num_grid_points[2]
            + x_ind * (num_grid_points[1] * num_grid_points[2])
        )

    edges = set()
    for i in range(num_grid_points[0]):
        for j in range(num_grid_points[1]):
            for k in range(num_grid_points[2]):
                xyz_ind = flatten_xyz_inds(i, j, k)

                # Generate the three edges to the next nodes along each axis
                # NOTE: no need to sort the edge tuple since the edge index always increases
                new_edges = []
                if i + 1 < num_grid_points[0]:
                    new_edges.append((xyz_ind, flatten_xyz_inds(i + 1, j, k)))
                if j + 1 < num_grid_points[1]:
                    new_edges.append((xyz_ind, flatten_xyz_inds(i, j + 1, k)))
                if k + 1 < num_grid_points[2]:
                    new_edges.append((xyz_ind, flatten_xyz_inds(i, j, k + 1)))
                edges.update(new_edges)

    return nodes_xyz, edges


def _find_points_within_mesh(
    points: np.ndarray,
    mesh: o3d.geometry.TriangleMesh,
    distance_threshold: float = 1.0,
    within_mesh_threshold: float = 0.2,
    kdtree_query_k: int = 5,
    kdtree_query_num_workers: int = 1,
) -> np.ndarray:
    """
    For an array of (N,3) points, find which points are considered "within"
    the mesh. Return this information as a (N,) binary mask.

    Note that this assumes that the mesh has reasonable vertex normals in order
    for the "within" criteria to be reasonable.

    Args:
        points: (N,3) array of points
        mesh: o3d.geometry.TriangleMesh
        distance_threshold: Distance threshold for a point to be considered
            "near" the mesh
        within_mesh_threshold: Threshold for a point to be considered "within"
            the mesh
        kdtree_query_k: Number of nearest neighbors to query for each point
        kdtree_query_num_workers: Number of workers to use for the kdtree query

    Returns:
        mask: (N,) binary mask of points which are considered within the mesh
    """
    verts = np.asarray(mesh.vertices)
    vert_normals = np.asarray(mesh.vertex_normals)

    verts_kd_tree = KDTree(verts)
    dd, ii = verts_kd_tree.query(  # dd: (N, K), ii: (N, K)
        points, k=kdtree_query_k, workers=kdtree_query_num_workers
    )

    # check points within distance to the mesh
    dist_mesh = np.mean(dd, axis=1)
    near_mesh = dist_mesh <= distance_threshold

    # check points within the mesh
    nearest_verts = verts[ii]  # (N, K, 3)
    nearest_vert_normals = vert_normals[ii]  # (N, K, 3)
    nearest_verts_to_nodes = np.expand_dims(points, axis=1) - nearest_verts  # (N, K, 3)

    dot_product = np.sum(
        nearest_vert_normals * nearest_verts_to_nodes, axis=2
    )  # (N, K)
    within_nearest_verts = (dot_product >= 0.0).astype(np.int8)
    within_vert = np.mean(within_nearest_verts, axis=1)  # (N,)
    within_mesh = within_vert > within_mesh_threshold

    return np.logical_and(near_mesh, within_mesh)


from tag_mapping.utils import o3d_check_lines_collision


def _find_mesh_intersecting_edges(
    mesh: o3d.geometry.TriangleMesh,
    nodes_xyz: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """
    Find which edges intersect with the mesh.

    Args:
        mesh: o3d.geometry.TriangleMesh
        nodes_xyz: (N,3) array of node coordinates
        edges: (E,2) array of edges as tuples of node indices

    Returns:
        intersects_mask: (E,) binary mask of edges which intersect with the mesh
    """
    rc_scene = o3d.t.geometry.RaycastingScene()
    _ = rc_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    intersects_mask = o3d_check_lines_collision(
        rc_scene, nodes_xyz[edges[:, 0]], nodes_xyz[edges[:, 1]]
    )

    return intersects_mask
