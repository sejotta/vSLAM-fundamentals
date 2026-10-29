"""Plotting and export helpers.

Visualisation is deliberately dependency-light: Matplotlib for static 3D
figures and OpenCV for match contact sheets.  No browser or notebook is
required, which keeps the demo reproducible on a headless machine.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _as_xyz(poses: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim == 3:
        return array[:, :3, 3]
    if array.ndim == 2 and array.shape[1] == 3:
        return array
    raise ValueError("poses must be (N, 4, 4) or (N, 3)")


def _set_equal_3d(ax, points: np.ndarray) -> None:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return
    minimum = pts.min(axis=0)
    maximum = pts.max(axis=0)
    centre = (minimum + maximum) / 2.0
    radius = float(np.max(maximum - minimum)) / 2.0
    radius = max(radius, 1e-6)
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_trajectory(
    ground_truth: Sequence[np.ndarray] | np.ndarray | None,
    estimated: Sequence[np.ndarray] | np.ndarray,
    output_path: str | Path,
    title: str = "Camera trajectory",
) -> None:
    """Save a 3D plot, optionally comparing against ground truth."""
    est = _as_xyz(estimated)
    gt = _as_xyz(ground_truth) if ground_truth is not None else None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    if gt is not None:
        ax.plot(
            gt[:, 0], gt[:, 1], gt[:, 2],
            color="#222222", linewidth=2.2, label="ground truth",
        )
        ax.scatter(
            gt[:, 0], gt[:, 1], gt[:, 2],
            color="#222222", s=18, depthshade=False,
        )
        if len(gt):
            ax.scatter(*gt[0], color="#2ca02c", s=80, marker="o", label="start")
            ax.scatter(*gt[-1], color="#d62728", s=80, marker="X", label="end")

    ax.plot(
        est[:, 0], est[:, 1], est[:, 2],
        color="#1f77b4", linewidth=1.6, linestyle="--", label="estimated",
    )
    ax.scatter(
        est[:, 0], est[:, 1], est[:, 2],
        c=np.arange(len(est)), cmap="viridis", s=28, depthshade=False,
    )

    combined = est if gt is None else np.vstack([gt, est])
    _set_equal_3d(ax, combined)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.view_init(elev=22, azim=-62)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_map(
    points: np.ndarray,
    colors: np.ndarray | Sequence[tuple[int, int, int]] | None,
    poses: Sequence[np.ndarray] | np.ndarray,
    output_path: str | Path,
    title: str = "Sparse 3D map",
    max_points: int = 20000,
) -> None:
    """Save a coloured point-cloud plot with the estimated trajectory."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) > max_points:
        indices = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[indices]
        if colors is not None:
            colors = np.asarray(colors)[indices]

    centers = _as_xyz(poses)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if colors is not None:
        rgb = np.asarray(colors, dtype=np.float64)
        if rgb.max(initial=0) > 1.0:
            rgb = rgb / 255.0
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=np.clip(rgb, 0.0, 1.0), s=3, alpha=0.85, depthshade=False,
        )
    else:
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c="#1f77b4", s=3, alpha=0.85, depthshade=False,
        )

    ax.plot(
        centers[:, 0], centers[:, 1], centers[:, 2],
        color="#d62728", linewidth=2.0, label="camera centres",
    )
    ax.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        color="#d62728", s=25, depthshade=False,
    )
    _set_equal_3d(ax, np.vstack([pts, centers]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.view_init(elev=20, azim=-62)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def draw_matches(
    image1: np.ndarray,
    image2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    output_path: str | Path,
    max_matches: int = 80,
) -> None:
    """Save a contact sheet of point correspondences (RGB input, RGB file)."""
    p1 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float32).reshape(-1, 2)
    count = min(len(p1), len(p2), max_matches)
    if count == 0:
        return

    keypoints1 = [cv2.KeyPoint(float(x), float(y), 3.0) for x, y in p1[:count]]
    keypoints2 = [cv2.KeyPoint(float(x), float(y), 3.0) for x, y in p2[:count]]
    matches = [cv2.DMatch(i, i, 0.0) for i in range(count)]
    canvas = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches,
        None,
        matchColor=(0, 220, 60),
        singlePointColor=(255, 90, 90),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_trajectory_csv(
    poses: Sequence[np.ndarray] | np.ndarray,
    frame_names: Sequence[str],
    output_path: str | Path,
) -> None:
    """Write camera centres and rotation matrices to a CSV file."""
    array = np.asarray(poses, dtype=np.float64)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame", "x", "y", "z",
                "r00", "r01", "r02",
                "r10", "r11", "r12",
                "r20", "r21", "r22",
            ]
        )
        for name, T in zip(frame_names, array):
            writer.writerow(
                [name, *T[:3, 3].tolist(), *T[:3, :3].ravel().tolist()]
            )
