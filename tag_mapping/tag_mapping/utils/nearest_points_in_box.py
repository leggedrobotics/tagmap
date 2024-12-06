import numpy as np
import cvxpy as cp

from typing import Tuple


def nearest_points_in_box(
    box_corners: np.ndarray,
    box_center: np.ndarray,
    points: np.ndarray,
    solve_kwargs=None,
) -> np.ndarray:
    """
    Computes the points that are closests to the given points, bounded
    within the box by solving a QP.

    Args:
        box_corners: (8,3) array of box corners in order defined in _box_hrep()
        box_center: (3,) array of center coordinate of the box
        points: (N,3) array of the given points
        solve_kwargs: Keyword arguments to pass to cp.Problem.solve(),
            e.g. the solver, verbose, etc.

    Returns:
        (N,3) array of the closest points
    """
    if solve_kwargs is None:
        solve_kwargs = {"verbose": False, "solver": cp.ECOS}

    box_A, box_b = _box_hrep(box_corners, box_center)

    N = points.shape[0]
    X = cp.Variable((3, N))
    objective = cp.Minimize(cp.sum_squares(X - points.T))

    # NOTE: use reshape to allow broadcasting of the inequality
    constraints = [box_A @ X <= (box_b + box_A @ box_center).reshape(-1, 1)]

    prob = cp.Problem(objective, constraints)
    prob.solve(**solve_kwargs)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError("Closest point QP did not reach optimal solution")

    return X.value.T


def _box_hrep(corners: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes H-rep arrays A and b such that a point p is in
    the box if A(p - center) <= b

    Args:
        corners: (8, 3) corners in the relative order outlined as follows:
            (4) +---------+. (5)
                | ` .     |  ` .
                | (0) +---+-----+ (1)
                |     |   |     |
            (7) +-----+---+. (6)|
                ` .   |     ` . |
                (3) ` +---------+ (2)

        center: (3,) center of the box

    Returns:
        A: (6, 3)
        b: (6,)
    """

    def face_hrep(c, vo, vx, vy):
        ex = vx - vo
        ey = vy - vo
        n = np.cross(ex, ey)
        n /= np.linalg.norm(n)
        d = np.dot(n, vo - c)
        return n, d

    face_hreps = [
        face_hrep(center, corners[1], corners[0], corners[2]),
        face_hrep(center, corners[4], corners[5], corners[7]),
        face_hrep(center, corners[2], corners[3], corners[6]),
        face_hrep(center, corners[3], corners[0], corners[7]),
        face_hrep(center, corners[0], corners[1], corners[4]),
        face_hrep(center, corners[1], corners[2], corners[5]),
    ]

    A = np.concatenate([n.reshape(1, -1) for n, _ in face_hreps], axis=0)
    b = np.array([d for _, d in face_hreps])

    return A, b
