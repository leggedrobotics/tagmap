import os
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, Union, Optional

import uuid
from tag_mapping import TagMap, TagMapEntry
from tag_mapping.models import ImageTagger
from tag_mapping.filtering import valid_depth_frame
from tag_mapping.datasets.matterport import (
    read_matterport_image_file,
    read_matterport_depth_file,
    read_matterport_pose_file,
    read_matterport_intrinsics_file,
    MatterportFilenameBridge,
)


def generate_tag_map_from_matterport_scan(
    params: Dict,
    tagging_model: ImageTagger,
    scan_dir: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Generate a tag map from a matterport scan.

    Args:
        params: Dictionary of parameters for tag map generation.
        tagging_model: The image tagging model which defines the method filtered_tag_image().
        scan_dir: Path to the matterport scan directory.
        output_dir: Path of the directory to save the tag map.
        logger: Logger to use, if None a logger will be created at debug level.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

    scan_name = os.path.basename(scan_dir)
    logger.info(f"generating tag map from matterport scan {scan_name}")

    images_dir = os.path.join(scan_dir, "undistorted_color_images")
    depths_dir = os.path.join(scan_dir, "undistorted_depth_images")
    poses_dir = os.path.join(scan_dir, "matterport_camera_poses")
    intrinsics_dir = os.path.join(scan_dir, "matterport_camera_intrinsics")
    logger.info(
        f"reading images, depth images, and poses from: {images_dir}\n{depths_dir}\n{poses_dir}"
    )

    # Averge intrinsics across all frames to get an estimate for the scan
    intrinsics = []
    for filename in os.listdir(intrinsics_dir):
        intrinsics_filepath = os.path.join(intrinsics_dir, filename)
        width, height, fx, fy, cx, cy, d = read_matterport_intrinsics_file(
            intrinsics_filepath
        )
        intrinsics.append([width, height, fx, fy])
    intrinsics = np.array(intrinsics)
    intrinsics = np.mean(intrinsics, axis=0)
    width, height, fx, fy = intrinsics
    logger.info(
        f"mean intrinsics over the scan: {width:.0f} {height:.0f} {fx:.2f} {fy:.2f}"
    )

    # Pack tag map metadata
    tag_map_metadata = {
        "scan_name": scan_name,
        "intrinsics": {
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "near_dist": params["matterport_viewpoint_near_dist"],
        },
        "tagging_model": tagging_model.__class__.__name__,
    }

    # Start tag map generation
    logger.info("starting tag map generation")
    tag_map = TagMap(metadata=tag_map_metadata)

    skipped_frames = []

    for image_filename in tqdm(os.listdir(images_dir)):
        filename_bridge = MatterportFilenameBridge.from_image_filename(image_filename)
        depth_filename = filename_bridge.depth_filename
        pose_filename = filename_bridge.pose_filename

        image = read_matterport_image_file(os.path.join(images_dir, image_filename))
        depth, depth_image = read_matterport_depth_file(
            os.path.join(depths_dir, depth_filename)
        )
        T_cam_to_world = read_matterport_pose_file(
            os.path.join(poses_dir, pose_filename)
        )

        # skip frames with invalid depth values
        if not valid_depth_frame(depth, **params["depth_filtering_params"]):
            skipped_frames.append((image_filename, depth_filename))
            continue

        # compute the tags and their confidences
        tags, confidences = tagging_model.filtered_tag_image(
            image, params=params["filtered_tagging_params"]
        )

        # information to store about the depth frame
        depth_percentiles = {
            str(q): dq
            for q, dq in zip(
                params["stored_depth_percentiles"],
                np.quantile(depth, params["stored_depth_percentiles"]),
            )
        }

        # pack data to store within a TagMapEntry and add it to the tag map
        entry_uuid = uuid.uuid4()
        entry = TagMapEntry(
            pose=T_cam_to_world,
            uuid=entry_uuid,
            extras={
                "depth_percentiles": depth_percentiles,
            },
        )
        tag_map.add_entry(entry)

        # add associated tags to the database
        for tag, conf in zip(tags, confidences):
            tag_map.add_tag(
                tag,
                entry_uuid,
                extras={},
            )

    logger.info(
        f"finished tag map generation, skipped {len(skipped_frames)} frames with invalid depth values"
    )

    # Save the tag map
    save_path = os.path.join(output_dir, f"{scan_name}.tagmap")
    tag_map.save(save_path)
    logger.info(f"saved tag map to {save_path}")
