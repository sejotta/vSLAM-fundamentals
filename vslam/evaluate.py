"""Trajectory evaluation utilities.

Monocular SLAM is only defined up to a global similarity transform (rotation,
translation and scale).  Before comparing an estimate with ground truth we
therefore align it with Umeyama's closed-form Sim(3) solution.  After that
alignment, absolute trajectory error (ATE) and relative pose error (RPE) are
meaningful.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry import rotation_error_deg


def umeyama_alignment(
    source: np.ndarray,
    target: np.ndarray,
    with_scale: bool = True,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Align ``source`` to ``target`` with a similarity transform.

    Returns ``(aligned, scale, R, t)`` such that
    ``aligned = scale * (R @ source.T).T + t``.
    """
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError("source and target must have the same shape")
    if len(src) < 3:
        raise ValueError("at least three points are required for alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    covariance = (dst_centered.T @ src_centered) / len(src)
    u, singular_values, vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt
    variance_src = np.mean(np.sum(src_centered**2, axis=1))
    if with_scale and variance_src > 1e-15:
        scale = float(
            np.sum(singular_values * np.diag(correction)) / variance_src
        )
    else:
        scale = 1.0

    translation = mu_dst - scale * rotation @ mu_src
    aligned = scale * (rotation @ src.T).T + translation
    return aligned, scale, rotation, translation


def trajectory_metrics(
    estimated_poses: Sequence[np.ndarray] | np.ndarray,
    ground_truth_poses: Sequence[np.ndarray] | np.ndarray,
) -> dict[str, float | int | np.ndarray]:
    """Compute ATE, rotation RPE and translation RPE after Sim(3) alignment."""
    estimated = np.asarray(estimated_poses, dtype=np.float64)
    truth = np.asarray(ground_truth_poses, dtype=np.float64)
    if estimated.ndim != 3 or truth.ndim != 3:
        raise ValueError("poses must be (N, 4, 4) arrays")
    count = min(len(estimated), len(truth))
    if count < 3:
        raise ValueError("at least three poses are required")
    estimated = estimated[:count]
    truth = truth[:count]

    est_xyz = estimated[:, :3, 3]
    gt_xyz = truth[:, :3, 3]
    aligned, scale, rotation, translation = umeyama_alignment(est_xyz, gt_xyz)
    errors = np.linalg.norm(aligned - gt_xyz, axis=1)

    rotation_errors = []
    translation_errors = []
    for index in range(count - 1):
        R_est = estimated[index, :3, :3].T @ estimated[index + 1, :3, :3]
        R_gt = truth[index, :3, :3].T @ truth[index + 1, :3, :3]
        rotation_errors.append(rotation_error_deg(R_est, R_gt))
        step_est = aligned[index + 1] - aligned[index]
        step_gt = gt_xyz[index + 1] - gt_xyz[index]
        translation_errors.append(float(np.linalg.norm(step_est - step_gt)))

    return {
        "poses": int(count),
        "ate_rmse": float(np.sqrt(np.mean(errors**2))),
        "ate_mean": float(np.mean(errors)),
        "ate_max": float(np.max(errors)),
        "scale": float(scale),
        "rpe_rotation_mean_deg": float(np.mean(rotation_errors)),
        "rpe_rotation_max_deg": float(np.max(rotation_errors)),
        "rpe_translation_mean": float(np.mean(translation_errors)),
        "aligned_xyz": aligned,
        "rotation": rotation,
        "translation": translation,
    }
