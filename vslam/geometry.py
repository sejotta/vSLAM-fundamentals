"""Multi-view geometry primitives.

All pose matrices are 4x4 homogeneous transforms.  The suffix convention is
important and is used consistently across the package:

``T_cw``
    World -> camera.  A world point ``X_w`` becomes ``X_c = T_cw @ X_w``.
    This is the matrix returned by OpenCV's PnP functions.

``T_wc``
    Camera -> world, i.e. the pose of the camera in the world.  It is the
    inverse of ``T_cw`` and its translation column is the camera centre.

For a pair of cameras, ``T_21`` maps camera-1 coordinates to camera-2
coordinates: ``X_2 = T_21 @ X_1``.  This matches ``cv2.recoverPose``.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Append a homogeneous coordinate to an ``(N, D)`` array."""
    pts = np.asarray(points, dtype=np.float64)
    return np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])


def invert_pose(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform without a generic matrix inverse."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def pose_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Build ``T_cw`` from an OpenCV rotation vector and translation."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def rvec_tvec_from_pose(T_cw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``T_cw`` into the representation used by OpenCV PnP."""
    rvec, _ = cv2.Rodrigues(np.asarray(T_cw, dtype=np.float64)[:3, :3])
    tvec = np.asarray(T_cw, dtype=np.float64)[:3, 3].reshape(3, 1)
    return rvec, tvec


def project_points(
    K: np.ndarray, T_cw: np.ndarray, points_world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points through a camera pose.

    Returns ``(uv, depth)`` where ``uv`` is ``(N, 2)`` and ``depth`` is the
    camera-frame ``z`` coordinate.
    """
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T_cw, dtype=np.float64)
    X = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    X_cam = (T[:3, :3] @ X.T + T[:3, 3:4]).T
    z = X_cam[:, 2]
    uv = np.full((X.shape[0], 2), np.nan, dtype=np.float64)
    front = z > 1e-12
    if np.any(front):
        homogeneous = (K @ X_cam[front].T).T
        uv[front] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return uv, z


def reprojection_errors(
    K: np.ndarray,
    T_cw: np.ndarray,
    points_world: np.ndarray,
    points_image: np.ndarray,
) -> np.ndarray:
    """Euclidean reprojection error for each 3D/2D correspondence."""
    uv, depth = project_points(K, T_cw, points_world)
    err = np.full(len(uv), np.inf, dtype=np.float64)
    front = depth > 1e-12
    if np.any(front):
        err[front] = np.linalg.norm(
            uv[front] - np.asarray(points_image, dtype=np.float64)[front], axis=1
        )
    return err


def fundamental_matrix(
    points1: np.ndarray,
    points2: np.ndarray,
    ransac_threshold: float = 0.8,
    confidence: float = 0.999,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Estimate ``F`` with the 8-point algorithm inside RANSAC.

    The relation is ``x2^T F x1 = 0``.  At least eight correspondences are
    required.  Returns ``(F, inlier_mask)``; ``F`` is ``None`` on failure.
    """
    p1 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float32).reshape(-1, 2)
    if len(p1) < 8 or len(p2) < 8:
        return None, np.zeros(len(p1), dtype=bool)

    F, mask = cv2.findFundamentalMat(
        p1,
        p2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=float(ransac_threshold),
        confidence=float(confidence),
    )
    if F is None or F.shape != (3, 3):
        return None, np.zeros(len(p1), dtype=bool)
    return F, np.asarray(mask).ravel().astype(bool)


def essential_from_fundamental(
    F: np.ndarray, K1: np.ndarray, K2: np.ndarray
) -> np.ndarray:
    """Convert a fundamental matrix to an essential matrix."""
    E = np.asarray(K2, dtype=np.float64).T @ np.asarray(F, dtype=np.float64)
    E = E @ np.asarray(K1, dtype=np.float64)
    norm = np.linalg.norm(E)
    return E / norm if norm > 0 else E


def relative_pose_from_essential(
    E: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover the relative pose with the cheirality check.

    Returns ``(R, t, mask)`` such that ``X_2 = R X_1 + t``.  OpenCV tries the
    four possible decompositions and keeps the one that places the
    triangulated points in front of both cameras.
    """
    p1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
    _, R, t, mask = cv2.recoverPose(
        np.asarray(E, dtype=np.float64), p1, p2, np.asarray(K, dtype=np.float64)
    )
    return R, t.reshape(3), np.asarray(mask).ravel().astype(bool)


def relative_pose_matrix(
    E: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(T_21, mask)`` with ``T_21`` a 4x4 transform."""
    R, t, mask = relative_pose_from_essential(E, points1, points2, K)
    T_21 = np.eye(4, dtype=np.float64)
    T_21[:3, :3] = R
    T_21[:3, 3] = t
    return T_21, mask


def triangulate_two_view(
    points1: np.ndarray,
    points2: np.ndarray,
    K1: np.ndarray,
    K2: np.ndarray,
    T_21: np.ndarray,
) -> np.ndarray:
    """Linear triangulation of matched points.

    Camera 1 is the origin.  ``T_21`` maps camera-1 coordinates to camera-2
    coordinates.  Returns an ``(N, 3)`` array in camera-1 coordinates.
    """
    p1 = np.asarray(points1, dtype=np.float64).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float64).reshape(-1, 2)
    P1 = np.asarray(K1, dtype=np.float64) @ np.hstack(
        [np.eye(3), np.zeros((3, 1))]
    )
    P2 = np.asarray(K2, dtype=np.float64) @ np.asarray(T_21, dtype=np.float64)[:3, :]
    X_h = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    denominator = X_h[3]
    denominator = np.where(np.abs(denominator) < 1e-12, np.nan, denominator)
    return (X_h[:3] / denominator).T


def triangulate_multiview(
    observations: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
) -> np.ndarray | None:
    """Triangulate one point from two or more ``(T_cw, uv)`` observations.

    This is the direct linear transform (DLT) written with the projection
    matrix rows.  It is used by the refinement pass, where a map point can be
    observed by more than two keyframes.
    """
    if len(observations) < 2:
        return None

    rows: list[np.ndarray] = []
    for T_cw, uv in observations:
        P = np.asarray(K, dtype=np.float64) @ np.asarray(T_cw, dtype=np.float64)[
            :3, :
        ]
        u, v = float(uv[0]), float(uv[1])
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])

    A = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(A)
    X_h = vh[-1]
    if abs(X_h[3]) < 1e-12:
        return None
    return X_h[:3] / X_h[3]


def pnp_pose(
    points3d: np.ndarray,
    points2d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None = None,
    ransac: bool = True,
    reprojection_threshold: float = 4.0,
    iterations: int = 200,
    confidence: float = 0.999,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Estimate ``T_cw`` from 3D-2D correspondences.

    Returns ``(T_cw, inlier_mask)``.  ``T_cw`` is ``None`` when the problem is
    under-determined or the solver fails.  At least six points are required
    for a stable PnP solution.
    """
    obj = np.ascontiguousarray(points3d, dtype=np.float64).reshape(-1, 1, 3)
    img = np.ascontiguousarray(points2d, dtype=np.float64).reshape(-1, 1, 2)
    if len(obj) < 6:
        return None, np.zeros(len(obj), dtype=bool)

    dist_coeffs = (
        np.zeros(5, dtype=np.float64)
        if dist is None
        else np.asarray(dist, dtype=np.float64).reshape(-1)
    )
    K = np.asarray(K, dtype=np.float64)

    inliers = None
    if ransac:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            K,
            dist_coeffs,
            iterationsCount=int(iterations),
            reprojectionError=float(reprojection_threshold),
            confidence=float(confidence),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            K,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    if not ok:
        return None, np.zeros(len(obj), dtype=bool)

    T_cw = pose_from_rvec_tvec(rvec, tvec)
    mask = np.zeros(len(obj), dtype=bool)
    if inliers is not None:
        mask[np.asarray(inliers).ravel().astype(int)] = True
    else:
        mask[:] = True
    return T_cw, mask


def reprojection_error_rmse(
    K: np.ndarray,
    T_cw: np.ndarray,
    points3d: np.ndarray,
    points2d: np.ndarray,
) -> float:
    """Root-mean-square reprojection error, ignoring points behind the camera."""
    err = reprojection_errors(K, T_cw, points3d, points2d)
    finite = np.isfinite(err)
    if not np.any(finite):
        return float("inf")
    return float(np.sqrt(np.mean(err[finite] ** 2)))


def triangulation_angle_deg(
    centre1: np.ndarray, centre2: np.ndarray, point3d: np.ndarray
) -> float:
    """Angle subtended by a 3D point at the two camera centres."""
    a = np.asarray(point3d, dtype=np.float64) - np.asarray(centre1, dtype=np.float64)
    b = np.asarray(point3d, dtype=np.float64) - np.asarray(centre2, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices."""
    R = np.asarray(R1, dtype=np.float64).T @ np.asarray(R2, dtype=np.float64)
    cosine = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
