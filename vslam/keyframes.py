"""Keyframe selection based on accumulated optical flow.

Using every video frame is wasteful: consecutive frames are almost identical
and triangulation from a tiny baseline is ill-conditioned.  Instead we keep a
frame whenever the tracked image motion since the last keyframe exceeds a
threshold.  The result is a small set of views with enough overlap and enough
parallax for triangulation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray:
    """Read an image as grayscale and fail with a useful message."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def mean_optical_flow(
    gray1: np.ndarray,
    gray2: np.ndarray,
    max_corners: int = 300,
    quality_level: float = 0.01,
    min_distance: float = 7.0,
) -> float:
    """Mean displacement of tracked corners between two grayscale frames."""
    corners = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=int(max_corners),
        qualityLevel=float(quality_level),
        minDistance=float(min_distance),
    )
    if corners is None or len(corners) == 0:
        return 0.0

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)
    if tracked is None or status is None:
        return 0.0
    good = status.ravel() == 1
    if int(good.sum()) < 5:
        return 0.0
    return float(np.linalg.norm(tracked[good] - corners[good], axis=1).mean())


def uniform_subset(indices: Sequence[int], count: int) -> list[int]:
    """Pick ``count`` entries evenly spread over ``indices``."""
    items = list(indices)
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items
    positions = np.linspace(0, len(items) - 1, count)
    chosen = sorted({int(round(p)) for p in positions})
    return [items[i] for i in chosen]


def select_keyframes(
    image_paths: Sequence[str | Path],
    translation_threshold: float = 25.0,
    max_keyframes: int = 10,
    max_corners: int = 300,
    quality_level: float = 0.01,
    min_distance: float = 7.0,
) -> list[int]:
    """Return the indices of the frames to use as keyframes.

    The first frame is always included.  A new candidate is added when the
    accumulated mean optical flow since the previous keyframe reaches
    ``translation_threshold`` pixels.  Candidates are then uniformly
    subsampled to at most ``max_keyframes``.

    Parameters
    ----------
    image_paths:
        Ordered sequence of image paths.
    translation_threshold:
        Accumulated mean pixel motion that triggers a keyframe.
    max_keyframes:
        Upper bound on the number of returned keyframes.
    """
    paths = list(image_paths)
    if not paths:
        return []
    if len(paths) == 1 or max_keyframes == 1:
        return [0]

    previous = read_gray(paths[0])
    candidates = [0]
    accumulated = 0.0

    for index in range(1, len(paths)):
        current = read_gray(paths[index])
        accumulated += mean_optical_flow(
            previous,
            current,
            max_corners=max_corners,
            quality_level=quality_level,
            min_distance=min_distance,
        )
        if accumulated >= translation_threshold:
            candidates.append(index)
            accumulated = 0.0
        previous = current

    if len(candidates) < 2:
        # No motion threshold was reached: fall back to uniform sampling.
        candidates = list(range(len(paths)))

    return uniform_subset(candidates, max_keyframes)
