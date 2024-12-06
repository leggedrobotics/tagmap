from typing import Union

import hdbscan
import numpy as np
import open3d as o3d


def cluster_points_dbscan(
    points: Union[np.ndarray, o3d.geometry.PointCloud], **dbscan_kwargs
) -> np.ndarray:
    """
    Cluster points using DBSCAN implementation from Open3D
    http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html#open3d.geometry.PointCloud.cluster_dbscan

    Args:
        points: (N, 3) array of points or Open3D point cloud
        **dbscan_kwargs: keyword arguments to pass to dbscan

    Returns:
        (N,) array of cluster labels
    """
    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif isinstance(points, o3d.geometry.PointCloud):
        pcd = points
    else:
        raise ValueError("points must be either an array or an Open3D point cloud")

    labels = np.array(pcd.cluster_dbscan(**dbscan_kwargs)).astype(np.int32)
    return labels


def cluster_points_hdbscan(
    points: Union[np.ndarray, o3d.geometry.PointCloud], **hdbscan_kwargs
) -> np.ndarray:
    """
    Cluster points using HDBSCAN implementation from hdbscan package
    https://github.com/scikit-learn-contrib/hdbscan

    Args:
        points: (N, 3) array of points or Open3D point cloud
        **hdbscan_kwargs: keyword arguments to pass to hdbscan

    Returns:
        (N,) array of cluster labels
    """
    if isinstance(points, np.ndarray):
        X = points
    elif isinstance(points, o3d.geometry.PointCloud):
        X = np.asarray(points.points)
    else:
        raise ValueError("points must be either an array or an Open3D point cloud")

    hdbscan_clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)

    labels = hdbscan_clusterer.fit_predict(X).astype(np.int32)
    return labels


def cluster_points(
    points: Union[np.ndarray, o3d.geometry.PointCloud],
    algorithm: str,
    **algorithm_kwargs,
) -> np.ndarray:
    """
    Cluster points using the specified algorithm

    Args:
        points: (N, 3) array of points or Open3D point cloud
        algorithm: algorithm to use for clustering
            one of 'dbscan' or 'hdbscan'
        **algorithm_kwargs: keyword arguments to pass to the clustering algorithm

    Returns:
        (N,) array of cluster labels
    """
    if algorithm == "dbscan":
        labels = cluster_points_dbscan(points, **algorithm_kwargs)
    elif algorithm == "hdbscan":
        labels = cluster_points_hdbscan(points, **algorithm_kwargs)
    else:
        raise ValueError(
            "Invalid algorithm: {}. Must be dbscan or hdbscan".format(algorithm)
        )
    return labels
