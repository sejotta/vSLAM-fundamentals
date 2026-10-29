"""Pinhole camera model and projection helpers.

Conventions
-----------
* Camera coordinates follow OpenCV: ``x`` right, ``y`` down, ``z`` forward.
* ``K`` is the 3x3 intrinsic matrix.
* ``dist`` stores OpenCV distortion coefficients in the order
  ``(k1, k2, p1, p2, k3)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    """A calibrated pinhole camera.

    Parameters
    ----------
    fx, fy, cx, cy:
        Focal lengths and principal point in pixels.
    width, height:
        Image size.  Zero is allowed when the size is not known in advance.
    dist:
        OpenCV distortion coefficients.  The default is a zero vector.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0
    dist: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        if dist.size not in (0, 4, 5, 8, 12, 14):
            raise ValueError(
                "dist must contain 0, 4, 5, 8, 12 or 14 coefficients"
            )
        object.__setattr__(self, "dist", dist)

    @property
    def K(self) -> np.ndarray:
        """The 3x3 intrinsic matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @property
    def image_size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.width, self.height

    def scaled(self, scale: float) -> "Camera":
        """Return a copy whose intrinsics refer to a resized image."""
        return Camera(
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
            width=int(round(self.width * scale)),
            height=int(round(self.height * scale)),
            dist=self.dist.copy(),
        )

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Undistort an ``(N, 2)`` array of pixel coordinates."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if self.dist.size == 0 or not np.any(self.dist):
            return pts.reshape(-1, 2).copy()
        out = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        return out.reshape(-1, 2)

    def project(self, points_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points to pixels.

        Parameters
        ----------
        points_cam:
            ``(N, 3)`` points expressed in camera coordinates.

        Returns
        -------
        uv : (N, 2) array
            Pixel coordinates.  Points with ``z <= 0`` are ``nan``.
        depth : (N,) array
            Camera-frame ``z`` coordinate.
        """
        pts = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = pts[:, 2]
        uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
        front = z > 1e-12
        if np.any(front):
            xy = pts[front, :2] / z[front, None]
            uv[front, 0] = self.fx * xy[:, 0] + self.cx
            uv[front, 1] = self.fy * xy[:, 1] + self.cy
        return uv, z
