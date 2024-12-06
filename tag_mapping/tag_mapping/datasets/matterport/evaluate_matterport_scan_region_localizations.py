import os
import pickle
import logging
from tqdm import tqdm
from typing import Dict, Optional, Union

import numpy as np
import open3d as o3d

from tag_mapping import TagMap
from tag_mapping.localization import tagmap_entries_to_viewpoints, localization_pipeline
from tag_mapping.evaluation import (
    LatticeNavigationGraph,
    assign_label_box_lattice_graph_nodes,
    assign_proposal_box_lattice_graph_nodes,
)
from tag_mapping.utils import get_box_corners
from tag_mapping.datasets.matterport import read_matterport_region_bounding_boxes
from tag_mapping.datasets.matterport import MP_REGION_RAM_TAGS_MAPPING


def evaluate_matterport_scan_region_localizations(
    params: Dict,
    scan_dir: Union[str, os.PathLike],
    tag_map_path: Union[str, os.PathLike],
    lattice_graph_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Evaluates the tag map region/room localizations on a matterport scan by comparing
    the localized regions against the labeled ground-truth bounding boxes.

    Evaluation outputs are saved to a pickle file in output_dir.

    Args:
        params: Dictionary of parameters for the evaluation.
        scan_dir: Path to the matterport scan directory.
            The basename of the scan_dir is used as the scan name.
        tag_map_path: Path to the tag map corresponding to the scan.
            The filename of the tag map must contain the scan name.
        lattice_graph_path: Path to the lattice graph corresponding to the scan.
            The filename of the lattice graph must contain the scan name.
        output_dir: Directory to save the evaluation outputs to.
        logger: Logger to use, if None a logger will be created at debug level.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

    scan_name = os.path.basename(scan_dir)
    logger.info(f"running evaluation on matterport scan {scan_name}")
    assert scan_name in os.path.basename(
        tag_map_path
    ), f"Tag map does not match scan {scan_name}. Tag map at {tag_map_path}"
    assert scan_name in os.path.basename(
        lattice_graph_path
    ), f"Lattice graph does not match scan {scan_name}. Lattice graph at {lattice_graph_path}."

    house_file_path = os.path.join(
        scan_dir, "house_segmentations", f"{scan_name}.house"
    )
    logger.info(f"loaded house file from {house_file_path}")
    ply_file_path = os.path.join(scan_dir, "house_segmentations", f"{scan_name}.ply")
    logger.info(f"loaded ply file from {ply_file_path}")

    # Load the labeled ground-truth bounding boxes
    label_gt_boxes = read_matterport_region_bounding_boxes(house_file_path)

    # Load the tag map
    tag_map = TagMap.load(tag_map_path)
    logger.info(f"loaded tag map from {tag_map_path}")

    # Read required metadata from the tag map
    tag_map_metadata = tag_map.metadata
    try:
        tagging_model = tag_map_metadata["tagging_model"]
    except KeyError:
        raise KeyError(f"tag map metadata does not contain the key 'tagging_model'")
    try:
        intrinsics = tag_map_metadata["intrinsics"]
    except KeyError:
        raise KeyError(f"tag map metadata does not contain the key 'intrinsics'")

    # Load the lattice graph
    lattice_graph = LatticeNavigationGraph.load(lattice_graph_path)
    rc_scene = o3d.t.geometry.RaycastingScene()
    rc_scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.io.read_triangle_mesh(ply_file_path)
        )
    )
    logger.info(f"loaded lattice graph from {lattice_graph_path}")

    if tagging_model == "RAMTagger":
        mp_region_tag_mapping = MP_REGION_RAM_TAGS_MAPPING
    elif tagging_model == "RAMPlusTagger":
        mp_region_tag_mapping = MP_REGION_RAM_TAGS_MAPPING
    else:
        raise ValueError(f"Unsupported tagging model: {tagging_model}")

    # Run localization for each label
    logger.info(f"started localization pipeline")
    label_proposals = {}
    for label in tqdm(label_gt_boxes.keys()):
        if label in params["label_params"]["blacklisted_labels"]:
            continue

        tags = mp_region_tag_mapping[label]

        p_boxes, p_box_confidences, p_box_tags = ([], [], [])
        for tag in tags:
            if tag not in tag_map:
                logger.debug(f"tag {tag} not in the tag map, skipping")
                continue

            entries = tag_map.query(tag)
            viewpoints = tagmap_entries_to_viewpoints(
                entries=entries,
                intrinsics=intrinsics,
                **params["viewpoint_kwargs"],
            )
            loc_output = localization_pipeline(
                viewpoints, **params["localization_kwargs"], verbose=False
            )

            for l, box in loc_output["level_bbxes"]:
                p_boxes.append(box)
                p_box_confidences.append(l)
                p_box_tags.append(tag)

        label_proposals[label] = {
            "boxes": p_boxes,
            "confidences": p_box_confidences,
            "tags": p_box_tags,
        }
    logger.info(f"finished running localization pipeline")

    # Assign nodes of the lattice navigation graph to the labeled and proposed boxes
    label_lattice_inds = {}
    for label, gt_boxes in label_gt_boxes.items():
        gt_boxes_lattice_inds = []
        for gt_box in gt_boxes:
            inds = assign_label_box_lattice_graph_nodes(
                lattice_graph,
                rc_scene,
                get_box_corners(gt_box),
                # NOTE: DISABLE inflation for region box label assignment since regions are usually large
                enable_inflation=False,
            )
            gt_boxes_lattice_inds.append(inds)
        label_lattice_inds[label] = gt_boxes_lattice_inds

    for label, proposals in label_proposals.items():
        p_boxes = proposals["boxes"]
        p_boxes_lattice_inds = []
        for p_box in p_boxes:
            inds = assign_proposal_box_lattice_graph_nodes(
                lattice_graph,
                rc_scene,
                get_box_corners(p_box),
            )
            p_boxes_lattice_inds.append(inds)
        proposals["lattice_inds"] = p_boxes_lattice_inds

    # Evaluate P2E metric for all proposals
    logger.info(f"started computing P2E metrics")
    for label, proposals in tqdm(label_proposals.items()):
        # get all lattice inds for all ground-truth boxes of this label
        all_gt_lattice_inds = set()
        for inds in label_lattice_inds[label]:
            all_gt_lattice_inds.update(inds)
        all_gt_lattice_inds = np.array(list(all_gt_lattice_inds))

        if len(all_gt_lattice_inds) == 0:
            logger.debug(f"no ground-truth lattice inds for label {label}, skipping it")
            continue

        proposals["metrics"] = []
        for p_lattice_inds in proposals["lattice_inds"]:
            p_lattice_inds = np.array(p_lattice_inds)

            if len(p_lattice_inds) == 0:
                logger.debug(
                    f"proposal with no assigned lattice nodes in {label}, setting P2E to np.nan"
                )
                proposals["metrics"].append(
                    {
                        "p2e": np.nan,
                    }
                )
                continue

            # query shortest path for all pairs of label and proposal lattice inds
            batch_p_inds, batch_l_inds = np.meshgrid(
                p_lattice_inds, all_gt_lattice_inds, indexing="ij"
            )
            batch_p_inds, batch_l_inds = (
                batch_p_inds.flatten(),
                batch_l_inds.flatten(),
            )

            all_pairs_spl = lattice_graph.batch_shortest_path_length(
                batch_p_inds, batch_l_inds
            )
            all_pairs_spl = all_pairs_spl.reshape(
                len(p_lattice_inds), len(all_gt_lattice_inds)
            )

            p_spl = all_pairs_spl.min(axis=1)  # min over all the ground-truth dim

            # compute mean shortest path length
            # NOTE: some paths lengths could be np.inf indicating it's not possible to reach any
            # ground-truth lattice node from this proposal node
            if np.isinf(p_spl).all():
                p2e = np.inf
            else:
                p2e = np.mean(p_spl[np.isfinite(p_spl)])

            # compute portion of the proposal lattice nodes that are directly at a ground-truth lattice node
            portion_at_gt = np.sum(p_spl == 0) / len(p_spl)

            proposals["metrics"].append(
                {
                    "p2e": p2e,
                }
            )

    logger.info(f"finished computing P2E metrics")

    # Evaluate E2P metric for all labels
    logger.info(f"started computing E2P metrics")
    label_gt_boxes_metrics = {}
    for label, gt_boxes_lattice_inds in label_lattice_inds.items():
        if label in params["label_params"]["blacklisted_labels"]:
            continue

        # get all lattice inds for all proposal boxes of this label
        all_p_lattice_inds = set()
        for inds in label_proposals[label]["lattice_inds"]:
            all_p_lattice_inds.update(inds)
        all_p_lattice_inds = np.array(list(all_p_lattice_inds))

        if len(all_p_lattice_inds) == 0:
            logger.debug(f"no proposal lattice inds for label {label}, skipping it")
            label_gt_boxes_metrics[label] = len(gt_boxes_lattice_inds) * [
                {
                    "e2p": np.nan,
                }
            ]
            continue

        label_gt_boxes_metrics[label] = []
        for gt_lattice_inds in gt_boxes_lattice_inds:
            gt_lattice_inds = np.array(gt_lattice_inds)

            if len(gt_lattice_inds) == 0:
                logger.debug(
                    f"ground-truth box with no assigned lattice nodes in {label}, setting E2P to np.nan"
                )
                label_gt_boxes_metrics[label].append(
                    {
                        "e2p": np.nan,
                    }
                )
                continue

            # query shortest path for all pairs of label and proposal lattice inds
            batch_p_inds, batch_l_inds = np.meshgrid(
                all_p_lattice_inds, gt_lattice_inds, indexing="ij"
            )
            batch_p_inds, batch_l_inds = (
                batch_p_inds.flatten(),
                batch_l_inds.flatten(),
            )

            all_pairs_spl = lattice_graph.batch_shortest_path_length(
                batch_p_inds, batch_l_inds
            )
            all_pairs_spl = all_pairs_spl.reshape(
                len(all_p_lattice_inds), len(gt_lattice_inds)
            )

            p_spl = all_pairs_spl.min(axis=0)  # min over all the proposal dim

            # compute min shortest path length
            # NOTE: some paths lengths could be np.inf indicating it's not possible to reach any
            # ground-truth lattice node from this proposal node
            if np.isinf(p_spl).all():
                e2p = np.inf
            else:
                e2p = np.mean(p_spl[np.isfinite(p_spl)])

            label_gt_boxes_metrics[label].append(
                {
                    "e2p": e2p,
                }
            )

    logger.info(f"finished computing E2P metrics")

    # this is a workaround since the box is an open3d type which isn't pickleable
    for label, proposals in label_proposals.items():
        boxes_corners = []
        for p_box in proposals["boxes"]:
            boxes_corners.append(get_box_corners(p_box))
        proposals["boxes_corners"] = boxes_corners
        del proposals["boxes"]

    # Save computed metrics
    save_filename = f"{scan_name}.pkl"
    save_path = os.path.join(output_dir, save_filename)
    out = {
        "scan_dir": scan_dir,
        "house_file_path": house_file_path,
        "ply_file_path": ply_file_path,
        "tag_map_path": tag_map_path,
        "lattice_graph_path": lattice_graph_path,
        "label_gt_boxes": label_gt_boxes,
        "label_lattice_inds": label_lattice_inds,
        "label_proposals": label_proposals,
        "label_gt_boxes_metrics": label_gt_boxes_metrics,
    }
    with open(save_path, "wb") as f:
        pickle.dump(out, f)
    logger.info(f"saved evaluation outputs to {save_path}")
