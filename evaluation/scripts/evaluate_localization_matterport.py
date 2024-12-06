import os
import argparse
import logging
from datetime import datetime

from tag_mapping.datasets.matterport.evaluate_matterport_scan_object_localizations import (
    evaluate_matterport_scan_object_localizations,
)
from tag_mapping.datasets.matterport.evaluate_matterport_scan_region_localizations import (
    evaluate_matterport_scan_region_localizations,
)
from tag_mapping.utils import load_yaml_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tag maps from Matterport scans"
    )
    parser.add_argument("--params_path", type=str, help="Path to params file")
    parser.add_argument("--tag_maps_dir", type=str, help="Path to tag map file")
    parser.add_argument(
        "--lattice_graphs_dir", type=str, help="Path to lattice graph file"
    )
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
        help="Scans to generate tag maps for. If not specified, all scans will in matterport_dir will be processed.",
    )
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # Read params
    params = load_yaml_params(args.params_path)

    label_type = params["label_params"]["type"]
    if label_type == "object":
        evaluate_matterport_scan_localization = (
            evaluate_matterport_scan_object_localizations
        )
    elif label_type == "region":
        evaluate_matterport_scan_localization = (
            evaluate_matterport_scan_region_localizations
        )
    else:
        raise ValueError(f"Invalid label type {params['label_params']['type']}")

    # Create output save directory
    output_name = (
        f"matterport_{label_type}_evaluation" if args.output_name == None else args.output_name
    )
    output_save_dir = os.path.join(
        args.output_dir,
        f"{output_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
    )
    os.makedirs(output_save_dir, exist_ok=True)
    logger.info(
        f"created matterport evaluation outputs save directory {output_save_dir}"
    )

    # Copy param file to output save dir
    os.system(f"cp {args.params_path} {output_save_dir}/_evaluation_params.yaml")

    scan_names = (
        args.scans if args.scans != None else sorted(os.listdir(args.matterport_dir))
    )
    for scan_name in scan_names:
        logger.info(f"\n\nrunning evaluation on scan {scan_name}")

        scan_dir = os.path.join(args.matterport_dir, f"{scan_name}")
        if not os.path.isdir(scan_dir):
            logger.warning(f"skipping due to non-existing scan directory {scan_dir}")
            continue

        tag_map_path = os.path.join(args.tag_maps_dir, f"{scan_name}.tagmap")
        if not os.path.isfile(tag_map_path):
            logger.warning(f"skipping due to non-existing tag map {tag_map_path}")
            continue

        lattice_graph_path = os.path.join(
            args.lattice_graphs_dir, f"{scan_name}_lattice_graph.pkl"
        )
        if not os.path.isfile(lattice_graph_path):
            logger.warning(
                f"skipping due to non-existing lattice graph {lattice_graph_path}"
            )
            continue

        try:
            evaluate_matterport_scan_localization(
                params=params,
                scan_dir=os.path.abspath(scan_dir),
                tag_map_path=os.path.abspath(tag_map_path),
                lattice_graph_path=os.path.abspath(lattice_graph_path),
                output_dir=output_save_dir,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"failed to generate tag map for scan {scan_name}")
            logger.error(e)
            continue
