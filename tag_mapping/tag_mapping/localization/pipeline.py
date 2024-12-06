import numpy as np
import open3d as o3d

from typing import Any, Callable, Dict, Iterable, Optional

from tag_mapping import TagMapEntry
from .clustering import cluster_points
from .viewpoint import Viewpoint
from .voxel_voting import grid_voxel_voting


def localization_pipeline(
    viewpoints: Iterable[Viewpoint],
    params: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Tag map localization pipeline.

    Args:
        viewpoints: Iterable of viewpoints
        params: Dictionary of parameters for the pipeline.

    Returns:
        Dictionary of results from the pipeline.
            "voxel_center_points": (N,3) array of voxel center points
            "voxel_scores": (N,) array of scores for each voxel
            "level_bbxes": Dictionary mapping clustering level to the bounding boxes
                at that level.
    """
    vv_params, cl_params = (params["voxel_voting"], params["clustering"])

    ### Voxel voting
    if vv_params["viewpoint_weight"] == None:
        viewpoint_weight = None
    elif vv_params["viewpoint_weight"] == "confidence":
        viewpoint_weight = np.array([vp.extras["confidence"] for vp in viewpoints])
    else:
        raise ValueError(
            'Invalid viewpoint_weight {}. Must be None or "confidence"'.format(
                vv_params["viewpoint_weight"]
            )
        )

    voxel_center_points, votes = grid_voxel_voting(
        viewpoints, vv_params["voxel_size"], viewpoint_weight
    )

    # handle case where voxel voting fills no voxels
    # then voxel_center_points and votes will be
    # (0,3) and (0,) respectively
    if voxel_center_points.shape[0] == 0:
        return {
            "voxel_center_points": voxel_center_points,
            "voxel_scores": votes,
            "level_bbxes": [],
        }

    if vv_params["scoring_method"] == "normalized_votes":
        voxel_scores = votes / np.max(votes)
        clustering_levels = cl_params["clustering_levels"]

        def score_to_votes(score):
            score = score * np.max(votes)
            if score == 0.0:
                return 1
            else:
                if score != int(score):
                    return int(np.ceil(score))
                else:
                    return int(score)

    elif vv_params["scoring_method"] == "votes":
        voxel_scores = votes
        voxel_levels = np.unique(votes)
        clustering_levels = range(1, voxel_levels.max() + 1)
        score_to_votes = lambda score: score
    else:
        raise ValueError(
            'Invalid scoring_method {}. Must be "normalized_votes" or "votes"'.format(
                vv_params["scoring_method"]
            )
        )

    ### Clustering
    if cl_params["algorithm"] == "dbscan":
        cluster_fn = lambda pcd: cluster_points(
            pcd, algorithm="dbscan", **cl_params["dbscan_kwargs"]
        )
    elif cl_params["algorithm"] == "hdbscan":
        cluster_fn = lambda pcd: cluster_points(
            pcd, algorithm="hdbscan", **cl_params["hdbscan_kwargs"]
        )
    else:
        raise ValueError(
            'Invalid algorithm {}. Must be "dbscan" or "hdbscan"'.format(
                cl_params["algorithm"]
            )
        )

    def bb_fn(pcd, bb_type):
        if bb_type == "axis_aligned":
            box = pcd.get_axis_aligned_bounding_box()

            # Pad zero dimensions to avoid zero-volume bounding boxes
            if box.volume() == 0.0:
                if verbose:
                    print(
                        "[warning]: bounding box with zero volume, padding zero length dimensions to voxel_size."
                    )
                min_bound = box.get_min_bound()
                max_bound = box.get_max_bound()
                zero_dims = np.where(min_bound == max_bound)[0]
                min_bound[zero_dims] -= vv_params["voxel_size"] / 2
                max_bound[zero_dims] += vv_params["voxel_size"] / 2
                box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

        elif bb_type == "oriented":
            raise NotImplementedError(
                "Oriented bounding boxes not implemented yet."
            )  # TODO: implement?

        else:
            raise ValueError(
                'Invalid bounding_box_type {}. Must be "axis_aligned" or "oriented"'.format(
                    bb_type
                )
            )
        assert box.volume() > 0.0
        return box

    level_bbxes = []
    for level in clustering_levels:
        level_pcd = o3d.geometry.PointCloud()
        level_pcd.points = o3d.utility.Vector3dVector(
            voxel_center_points[voxel_scores >= level]
        )

        if len(level_pcd.points) == 0:
            if verbose:
                print(
                    "[warning]: no more points past level {}, stopping.".format(level)
                )
            break

        cluster_labels = cluster_fn(level_pcd)

        for i in range(cluster_labels.max() + 1):
            cluster_inds = np.where(cluster_labels == i)[0]
            cluster_pcd = level_pcd.select_by_index(cluster_inds)

            cluster_box = bb_fn(cluster_pcd, bb_type=cl_params["bounding_box_type"])

            level_bbxes.append((level, cluster_box))

    ## Apply non-max suppression to the clustered regions
    level_bbxes = sorted(level_bbxes, key=lambda x: x[0], reverse=True)
    remove_box = len(level_bbxes) * [False]
    for i in range(len(level_bbxes) - 1):
        l1, p1 = level_bbxes[i]
        for j in range(i + 1, len(level_bbxes)):
            # skip if p2 already marked for removal
            if remove_box[j]:
                continue

            l2, p2 = level_bbxes[j]

            if l1 == l2:
                continue

            p1_in_p2 = _box_contains_box(p2, p1)
            if p1_in_p2:
                remove_box[j] = True

    level_bbxes = [lb for i, lb in enumerate(level_bbxes) if not remove_box[i]]

    # Map scores used for clustering back to the more easily interpretable votes
    level_bbxes = [(score_to_votes(l), b) for l, b in level_bbxes]

    return {
        "voxel_center_points": voxel_center_points,
        "voxel_scores": voxel_scores,
        "level_bbxes": level_bbxes,  # TODO return instead a list of scored bounding boxes
    }


def tagmap_entries_to_viewpoints(
    entries: Iterable[TagMapEntry],
    intrinsics: Dict[str, Any],
    store_extras_keys: Iterable[str] = [],
    far_dist_fn: Optional[Callable[[TagMapEntry], float]] = None,
    near_dist_fn: Optional[Callable[[TagMapEntry], float]] = None,
) -> Iterable[Viewpoint]:
    """
    Helper function to convert an iterable of tag map entries their corresponding
    viewpoints.

    Args:
        entries: Iterable of entries from a tag map.
        intrinsics: Dictionary of camera intrinsics, must define
            ["width", "height", "fx", "fy"]
        store_extras_keys: Iterable of keys of the entry extras to store in
            the viewpoint extras.
        far_dist_fn: Function to compute the viewpoint's far distance from a query entry.
            If None, the viewpoint's far distance is set to intrinsics["far_dist"]
        near_dist_fn: Function to compute the viewpoint's near distance from a query entry
            If None, the viewpoint's near distance is set to intrinsics["near_dist"]
    """

    if far_dist_fn == None:
        far_dist_fn = lambda entry: intrinsics["far_dist"]

    if near_dist_fn == None:
        near_dist_fn = lambda entry: intrinsics["near_dist"]

    viewpoints = []
    for entry in entries:
        try:
            far_dist = far_dist_fn(entry)
        except Exception as e:
            print("Error in far_dist_fn, using value in intrinsics: {}.".format(e))

        try:
            near_dist = near_dist_fn(entry)
        except Exception as e:
            print("Error in near_dist_fn, using value in intrinsics: {}.".format(e))

        extras = {k: v for k, v in entry.extras.items() if k in store_extras_keys}

        if far_dist <= near_dist:
            # skip creating viewpoint if far_dist <= near_dist
            continue

        vp = Viewpoint.from_intrinsics(
            extrinsic_matrix=entry.pose,
            width=intrinsics["width"],
            height=intrinsics["height"],
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            near_dist=near_dist,
            far_dist=far_dist,
            extras=extras,
        )
        viewpoints.append(vp)

    return viewpoints


def _box_contains_box(box1, box2):
    """
    Helper function that returns True if box1 wholly contains box2.
    """
    # TODO implement this for other box types
    if (
        type(box1) != o3d.geometry.AxisAlignedBoundingBox
        or type(box2) != o3d.geometry.AxisAlignedBoundingBox
    ):
        raise NotImplementedError(
            "Unsupported box type, must be AxisAlignedBoundingBox"
        )

    box1_min_bound, box1_max_bound = (box1.get_min_bound(), box1.get_max_bound())
    box2_min_bound, box2_max_bound = (box2.get_min_bound(), box2.get_max_bound())
    return np.all(box1_min_bound <= box2_min_bound) and np.all(
        box1_max_bound >= box2_max_bound
    )
