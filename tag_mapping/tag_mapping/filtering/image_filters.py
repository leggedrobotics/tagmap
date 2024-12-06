import numpy as np

from typing import Iterable, Tuple, Optional


def valid_depth_frame(
    depth_frame: np.ndarray,
    mean_threshold: Optional[float] = None,
    quantile_thresholds: Optional[Iterable[Tuple[float, float]]] = None,
) -> bool:
    """
    Used to filter out frames of up-close views that are unlikely to be informative.

    Args:
        depth_frame: Depth frame to check.
        mean_threshold: minimum threshold on mean of the depth frame.
            Set to None to skip mean threshold check.
        quantile_thresholds: list of tuples of quantiles their minimum depth thresholds.
            Set to None to skip quantile threshold check.

    Returns:
        True if the depth frame is valid, False otherwise.
    """

    valid_depths_mask = np.logical_and(~np.isnan(depth_frame), ~np.isinf(depth_frame))

    if not np.any(valid_depths_mask):
        return False

    if mean_threshold != None:
        if np.mean(depth_frame[valid_depths_mask]) < mean_threshold:
            return False

    if quantile_thresholds != None:
        quantiles = np.quantile(
            depth_frame[valid_depths_mask], [q for q, _ in quantile_thresholds]
        )
        if np.any(quantiles < [thresh for _, thresh in quantile_thresholds]):
            return False

    return True
