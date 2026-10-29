"""Unit tests for the geometry core.

Run with::

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from vslam.evaluate import umeyama_alignment
from vslam.geometry import (
    essential_from_fundamental,
    fundamental_matrix,
    pnp_pose,
    project_points,
    relative_pose_matrix,
    rotation_error_deg,
    triangulate_two_view,
)
from vslam.homography import estimate_homography


def _rotation_y(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )


class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.K = np.array(
            [[650.0, 0.0, 320.0], [0.0, 650.0, 240.0], [0.0, 0.0, 1.0]]
        )
        rng = np.random.default_rng(3)
        self.points = rng.uniform(-2.0, 2.0, size=(200, 3))
        self.points[:, 2] = rng.uniform(3.0, 8.0, size=200)

        self.R = _rotation_y(8.0)
        self.t = np.array([0.5, 0.05, 0.1])
        self.T_21 = np.eye(4)
        self.T_21[:3, :3] = self.R
        self.T_21[:3, 3] = self.t

    def _project_pair(self):
        P0 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ np.hstack([self.R, self.t[:, None]])
        homogeneous = np.hstack(
            [self.points, np.ones((len(self.points), 1))]
        )
        p0 = (P0 @ homogeneous.T).T
        p1 = (P1 @ homogeneous.T).T
        return p0[:, :2] / p0[:, 2:3], p1[:, :2] / p1[:, 2:3]

    def test_triangulation_recovers_3d_points(self) -> None:
        points0, points1 = self._project_pair()
        recovered = triangulate_two_view(
            points0, points1, self.K, self.K, self.T_21
        )
        error = np.linalg.norm(recovered - self.points, axis=1)
        self.assertLess(float(np.mean(error)), 1e-6)

    def test_essential_matrix_recovers_pose(self) -> None:
        points0, points1 = self._project_pair()
        F, mask = fundamental_matrix(points0, points1, ransac_threshold=0.5)
        self.assertIsNotNone(F)
        self.assertGreater(int(mask.sum()), 180)

        E = essential_from_fundamental(F, self.K, self.K)
        recovered, pose_mask = relative_pose_matrix(
            E, points0, points1, self.K
        )
        self.assertGreater(int(pose_mask.sum()), 180)
        self.assertLess(
            rotation_error_deg(self.R, recovered[:3, :3]), 0.05
        )
        cosine = float(
            np.dot(
                self.t / np.linalg.norm(self.t),
                recovered[:3, 3] / np.linalg.norm(recovered[:3, 3]),
            )
        )
        self.assertGreater(abs(cosine), 0.999)

    def test_pnp_recovers_pose(self) -> None:
        rvec, _ = cv2.Rodrigues(self.R)
        tvec = self.t.reshape(3, 1)
        image_points, _ = cv2.projectPoints(
            self.points.astype(np.float64),
            rvec,
            tvec,
            self.K,
            np.zeros(5),
        )
        T_cw, inliers = pnp_pose(
            self.points,
            image_points.reshape(-1, 2),
            self.K,
            np.zeros(5),
        )
        self.assertIsNotNone(T_cw)
        self.assertGreater(int(inliers.sum()), 190)
        self.assertLess(
            rotation_error_deg(self.R, T_cw[:3, :3]), 0.01
        )
        self.assertLess(
            float(np.linalg.norm(T_cw[:3, 3] - self.t)), 0.01
        )

    def test_umeyama_recovers_similarity(self) -> None:
        rng = np.random.default_rng(9)
        source = rng.normal(size=(30, 3))
        rotation = _rotation_y(33.0)
        scale = 2.4
        translation = np.array([1.0, -0.4, 0.7])
        target = scale * (rotation @ source.T).T + translation

        aligned, recovered_scale, recovered_R, recovered_t = (
            umeyama_alignment(source, target)
        )
        self.assertAlmostEqual(recovered_scale, scale, places=6)
        self.assertLess(rotation_error_deg(rotation, recovered_R), 1e-5)
        self.assertLess(
            float(np.linalg.norm(recovered_t - translation)), 1e-5
        )
        self.assertLess(
            float(np.max(np.linalg.norm(aligned - target, axis=1))), 1e-5
        )

    def test_homography_recovers_planar_mapping(self) -> None:
        rng = np.random.default_rng(12)
        points = rng.uniform([0, 0], [640, 480], size=(80, 2))
        H = np.array(
            [[0.95, 0.08, 20.0], [-0.05, 1.02, -10.0], [0.0001, 0.0002, 1.0]]
        )
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        mapped = (H @ homogeneous.T).T
        mapped = mapped[:, :2] / mapped[:, 2:3]

        estimated, mask = estimate_homography(points, mapped)
        self.assertIsNotNone(estimated)
        self.assertGreater(int(mask.sum()), 75)

        predicted = (estimated @ homogeneous.T).T
        predicted = predicted[:, :2] / predicted[:, 2:3]
        error = np.linalg.norm(predicted - mapped, axis=1)
        self.assertLess(float(np.mean(error)), 1e-3)


if __name__ == "__main__":
    unittest.main()
