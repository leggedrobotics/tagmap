import os
import argparse
import logging
from datetime import datetime

from tag_mapping.models import RAMTagger, RAMPlusTagger

from tag_mapping.datasets.matterport.generate_tag_map_from_matterport_scan import (
    generate_tag_map_from_matterport_scan,
)
from tag_mapping.utils import load_yaml_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate tag maps from Matterport scans"
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
        help="Scans to generate tag maps for. If not specified, all scans will in matterport_dir will be processed.",
    )
    args = parser.parse_args()

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    # Read params
    params = load_yaml_params(args.params_path)
    model_params = params["model_params"]
    tag_map_generation_params = params["tag_map_generation_params"]

    # Create output save directory
    output_name = (
        "matterport_tag_maps" if args.output_name == None else args.output_name
    )
    output_save_dir = os.path.join(
        args.output_dir,
        f"{output_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
    )
    os.makedirs(output_save_dir, exist_ok=True)
    logger.info(f"created matterport tag maps output save directory {output_save_dir}")

    # Load tagging model
    if model_params["model"] == "ram":
        tagging_model = RAMTagger(
            config=model_params["model_config"],
        )
    elif model_params["model"] == "ram_plus":
        tagging_model = RAMPlusTagger(
            config=model_params["model_config"],
        )
    else:
        raise ValueError(f"Unsupported model type {model_params['model']}")

    # Copy param file to output save dir
    os.system(f"cp {args.params_path} {output_save_dir}/_gen_params.yaml")

    # Generate tag maps for each scan
    scan_names = (
        args.scans if args.scans != None else sorted(os.listdir(args.matterport_dir))
    )
    for scan_name in scan_names:
        scan_dir = os.path.join(args.matterport_dir, f"{scan_name}")

        if not os.path.isdir(scan_dir):
            logger.warning(f"skipping non-existing scan directory {scan_dir}")
            continue

        try:
            generate_tag_map_from_matterport_scan(
                params=tag_map_generation_params,
                tagging_model=tagging_model,
                scan_dir=scan_dir,
                output_dir=output_save_dir,
                logger=logger,
            )
        except Exception as e:
            logger.error(f"failed to generate tag map for scan {scan_name}")
            logger.error(e)
            continue
