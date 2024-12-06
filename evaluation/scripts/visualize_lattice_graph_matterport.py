import os
import argparse

import open3d as o3d

from tag_mapping.evaluation import LatticeNavigationGraph


def load_and_visualize_lattice_graph(
    lattice_graph_path,
    matterport_scans_dir,
):
    scan_name = os.path.basename(lattice_graph_path).split('_')[0]
    ply_path = os.path.join(
        matterport_scans_dir, f"{scan_name}/house_segmentations/{scan_name}.ply"
    )
    mesh = o3d.io.read_triangle_mesh(ply_path)
    lattice_graph = LatticeNavigationGraph.load(lattice_graph_path)
    o3d.visualization.draw_geometries(
        [
            mesh,
            lattice_graph.o3d_nodes_pointcloud,
            lattice_graph.o3d_edges_lineset,
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lattice_graph_path",
        type=str,
        required=True,
        help="Path to the lattice graph file",
    )
    parser.add_argument(
        "--matterport_dir", type=str, help="Path to directory of matterport scans"
    )
    args = parser.parse_args()

    load_and_visualize_lattice_graph(
        args.lattice_graph_path,
        args.matterport_dir,
    )
