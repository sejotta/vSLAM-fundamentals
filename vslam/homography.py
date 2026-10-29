"""Planar homography estimation and simple image mosaicking.

A homography is the right model when the scene is planar or the camera only
rotates.  It is included here as the planar counterpart to the epipolar
pipeline: the same feature matches feed either ``findHomography`` or
``findFundamentalMat``.
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_homography(
    points1: np.ndarray,
    points2: np.ndarray,
    ransac_threshold: float = 5.0,
    confidence: float = 0.999,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Estimate ``H`` such that ``x2 ~ H x1``.

    Returns ``(H, inlier_mask)``; ``H`` is ``None`` when fewer than four
    correspondences are available or the estimator fails.
    """
    p1 = np.asarray(points1, dtype=np.float32).reshape(-1, 2)
    p2 = np.asarray(points2, dtype=np.float32).reshape(-1, 2)
    if len(p1) < 4 or len(p2) < 4:
        return None, np.zeros(len(p1), dtype=bool)

    H, mask = cv2.findHomography(
        p1,
        p2,
        cv2.RANSAC,
        float(ransac_threshold),
        confidence=float(confidence),
    )
    if H is None or H.shape != (3, 3):
        return None, np.zeros(len(p1), dtype=bool)
    return H, np.asarray(mask).ravel().astype(bool)


def warp_pair(
    image1: np.ndarray,
    image2: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Warp ``image1`` into the plane of ``image2`` and merge both.

    ``H`` maps image-1 pixels to image-2 pixels.  The output canvas is large
    enough to contain both images, and ``image2`` is copied on top with a
    clean (non-blended) seam.  Empty regions are black.
    """
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    corners1 = np.float32(
        [[0, 0], [w1, 0], [w1, h1], [0, h1]]
    ).reshape(-1, 1, 2)
    corners1_warped = cv2.perspectiveTransform(corners1, H).reshape(-1, 2)
    corners2 = np.float32(
        [[0, 0], [w2, 0], [w2, h2], [0, h2]]
    ).reshape(-1, 2)

    all_corners = np.vstack([corners1_warped, corners2])
    x_min, y_min = all_corners.min(axis=0)
    x_max, y_max = all_corners.max(axis=0)

    offset_x = int(max(0, np.floor(-x_min)))
    offset_y = int(max(0, np.floor(-y_min)))
    canvas_w = int(np.ceil(x_max + offset_x))
    canvas_h = int(np.ceil(y_max + offset_y))

    translation = np.array(
        [[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]]
    )
    H_offset = translation @ np.asarray(H, dtype=np.float64)

    canvas = cv2.warpPerspective(
        image1,
        H_offset,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    canvas[
        offset_y : offset_y + h2,
        offset_x : offset_x + w2,
    ] = image2
    return canvas


def crop_black_borders(image: np.ndarray) -> np.ndarray:
    """Remove empty black margins left by ``warpPerspective``."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(threshold)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y : y + h, x : x + w]
