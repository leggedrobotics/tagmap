import os
import argparse
import logging
from datetime import datetime

import open3d as o3d

from tag_mapping.evaluation import create_lattice_navigation_graph
from tag_mapping.utils import load_yaml_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate lattice navigation graphs from Matterport scans"
    )
    parser.add_argument("--params_path", type=str, help="Path to params file")
    parser.add_argument("--output_dir", type=str, help="Path to output directory")
    parser.add_argument(
        "--output_name", type=str, help="Name of evaluation output directory"
    )
    parser.add_argument(
        "--matterport_dir", type=str, help="Path to directory of matterport scans"
    )
    parser.add_argument(
        "--scans",
        nargs="+",
        help="Scans to generate lattice navigation graphs for. If not specified, all scans will in matterport_dir will be processed.",
    )
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # Read params
    params = load_yaml_params(args.params_path)

    # Create output save directory
    output_name = (
        "matterport_lattice_graphs" if args.output_name == None else args.output_name
    )
    output_save_dir = os.path.join(
        args.output_dir,
        f"{output_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
    )
    os.makedirs(output_save_dir, exist_ok=True)
    logger.info(
        f"created matterport lattice graphs output save directory {output_save_dir}"
    )

    # Copy param file to output save dir
    os.system(f"cp {args.params_path} {output_save_dir}/_gen_params.yaml")

    # Generate lattice navigation graph for each scan
    scan_names = (
        args.scans if args.scans != None else sorted(os.listdir(args.matterport_dir))
    )
    for scan_name in scan_names:
        logger.info(f"\n\ncreating lattice graph for scan {scan_name}")
        scan_dir = os.path.join(args.matterport_dir, f"{scan_name}")
        ply_file_path = os.path.join(
            scan_dir, "house_segmentations", f"{scan_name}.ply"
        )
        mesh = o3d.io.read_triangle_mesh(ply_file_path)

        try:
            lattice_graph = create_lattice_navigation_graph(
                mesh,
                params=params["lattice_graph_creation_params"],
                print_progress=True,
            )
        except Exception as e:
            logger.error(f"failed to generate tag map for scan {scan_name}")
            logger.error(e)
            continue

        save_path = os.path.join(output_save_dir, f"{scan_name}_lattice_graph.pkl")
        lattice_graph.save(save_path)
        logger.info(f"saved lattice graph to {save_path}")
