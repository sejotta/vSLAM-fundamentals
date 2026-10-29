"""Incremental monocular visual SLAM.

The pipeline follows the classical structure of a visual odometry front-end:

1. pick keyframes with enough baseline,
2. match features between consecutive keyframes,
3. estimate the fundamental matrix with RANSAC,
4. turn it into an essential matrix and recover the relative pose,
5. localise the new keyframe with PnP against the existing map,
6. triangulate matches that do not have a 3D point yet,
7. refine poses and points with an alternating bundle-adjustment pass.

The map is a sparse coloured point cloud.  No loop closure, relocalisation or
IMU fusion is implemented; this is a compact, readable foundation rather than
a production SLAM system.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from .camera import Camera
from .features import Features, Matcher
from .geometry import (
    essential_from_fundamental,
    fundamental_matrix,
    invert_pose,
    pnp_pose,
    pose_from_rvec_tvec,
    project_points,
    relative_pose_matrix,
    rvec_tvec_from_pose,
    triangulate_multiview,
    triangulate_two_view,
    triangulation_angle_deg,
)


@dataclass
class SlamConfig:
    """Tunable parameters for :class:`IncrementalSlam`."""

    f_ransac_threshold: float = 0.8
    f_confidence: float = 0.999
    min_matches: int = 12
    min_init_inliers: int = 20
    min_init_points: int = 25
    min_depth: float = 0.05
    max_depth: float = 50.0
    max_reproj_error: float = 2.5
    min_triangulation_angle: float = 1.0
    pnp_reprojection_threshold: float = 4.0
    min_pnp_points: int = 6
    refine_iterations: int = 4
    refine_every: int = 2
    verbose: bool = True


@dataclass
class SlamResult:
    """Output of a SLAM run."""

    poses: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    keyframe_indices: list[int]
    stats: dict[str, float | int]
    observations: list[tuple[int, int, np.ndarray]] = field(
        default_factory=list, repr=False
    )


class IncrementalSlam:
    """Sparse feature-based visual SLAM front-end."""

    def __init__(
        self,
        camera: Camera,
        matcher: Matcher,
        config: SlamConfig | None = None,
    ) -> None:
        self.camera = camera
        self.matcher = matcher
        self.config = config or SlamConfig()

        self.images: list[np.ndarray] = []
        self.features: list[Features] = []
        self.poses: list[np.ndarray] = []  # T_wc, camera -> world
        self.points: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []
        self.observations: list[tuple[int, int, np.ndarray]] = []
        self.kp_to_point: dict[tuple[int, int], int] = {}
        self.stats: dict[str, float | int] = {}
        self._init_point_count = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(
        self,
        images: Sequence[np.ndarray],
        keyframe_indices: Sequence[int] | None = None,
    ) -> SlamResult:
        """Run the pipeline on a sequence of RGB images.

        ``images`` must already be the selected keyframes, in temporal order.
        ``keyframe_indices`` are the original indices of those frames and are
        only used for reporting.
        """
        if len(images) < 2:
            raise ValueError("at least two keyframe images are required")

        start = time.perf_counter()
        self.images = [np.asarray(image) for image in images]
        self.features = [self.matcher.detect(image) for image in self.images]
        indices = (
            list(range(len(images)))
            if keyframe_indices is None
            else [int(i) for i in keyframe_indices]
        )
        if len(indices) != len(images):
            raise ValueError("keyframe_indices must match the number of images")

        if self.config.verbose:
            print(f"[slam] features per keyframe: "
                  f"{[len(f.points) for f in self.features]}")

        self._initialize()

        pair_stats: list[dict[str, int]] = []
        for local_index in range(2, len(self.images)):
            stats = self._add_keyframe(local_index - 1, local_index)
            pair_stats.append(stats)
            should_refine = (
                self.config.refine_iterations > 0
                and self.config.refine_every > 0
                and (local_index + 1) % self.config.refine_every == 0
            )
            if should_refine:
                self._refine(self.config.refine_iterations)
                if self.config.verbose:
                    print(
                        f"[slam] keyframe {local_index + 1}/"
                        f"{len(self.images)}  "
                        f"points={len(self.points)}  "
                        f"rms={self._rms_reprojection():.2f}px"
                    )

        if self.config.refine_iterations > 0:
            self._refine(self.config.refine_iterations)

        elapsed = time.perf_counter() - start
        self.stats = {
            "keyframes": len(self.images),
            "map_points": len(self.points),
            "observations": len(self.observations),
            "init_matches": int(self.stats.get("init_matches", 0)),
            "init_points": int(self._init_point_count),
            "seconds": elapsed,
            "rms_reprojection_px": self._rms_reprojection(),
            "mean_matches": float(
                np.mean([s["matches"] for s in pair_stats])
            )
            if pair_stats
            else 0.0,
            "mean_inliers": float(
                np.mean([s["inliers"] for s in pair_stats])
            )
            if pair_stats
            else 0.0,
            "mean_pnp_points": float(
                np.mean([s["pnp"] for s in pair_stats])
            )
            if pair_stats
            else 0.0,
            "mean_new_points": float(
                np.mean([s["new_points"] for s in pair_stats])
            )
            if pair_stats
            else 0.0,
        }
        if self.config.verbose:
            print(
                f"[slam] done: {self.stats['keyframes']} keyframes, "
                f"{self.stats['map_points']} map points, "
                f"{self.stats['rms_reprojection_px']:.2f} px RMS, "
                f"{self.stats['seconds']:.1f} s"
            )

        colors = (
            np.asarray(self.colors, dtype=np.float64)
            if self.colors
            else np.empty((0, 3), dtype=np.float64)
        )
        return SlamResult(
            poses=np.asarray(self.poses, dtype=np.float64),
            points=np.asarray(self.points, dtype=np.float64).reshape(-1, 3),
            colors=colors,
            keyframe_indices=indices,
            stats=self.stats,
            observations=list(self.observations),
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        config = self.config
        features_a = self.features[0]
        features_b = self.features[1]
        matches = self.matcher.match(
            features_a,
            features_b,
            self.images[0],
            self.images[1],
        )
        if len(matches) < config.min_matches:
            raise RuntimeError(
                "initialisation failed: not enough feature matches "
                f"({len(matches)} < {config.min_matches})"
            )

        points_a = features_a.points[matches[:, 0]]
        points_b = features_b.points[matches[:, 1]]
        F, mask_f = fundamental_matrix(
            points_a,
            points_b,
            ransac_threshold=config.f_ransac_threshold,
            confidence=config.f_confidence,
        )
        if F is None or int(mask_f.sum()) < config.min_init_inliers:
            raise RuntimeError(
                "initialisation failed: RANSAC could not recover a stable "
                f"fundamental matrix ({int(mask_f.sum())} inliers)"
            )

        points_a = points_a[mask_f]
        points_b = points_b[mask_f]
        matches = matches[mask_f]

        E = essential_from_fundamental(
            F, self.camera.K, self.camera.K
        )
        T_21, mask_pose = relative_pose_matrix(
            E, points_a, points_b, self.camera.K
        )
        if int(mask_pose.sum()) < config.min_init_points:
            raise RuntimeError(
                "initialisation failed: cheirality check kept too few points "
                f"({int(mask_pose.sum())})"
            )

        points_a = points_a[mask_pose]
        points_b = points_b[mask_pose]
        matches = matches[mask_pose]

        points_3d = triangulate_two_view(
            points_a,
            points_b,
            self.camera.K,
            self.camera.K,
            T_21,
        )
        T_cw_0 = np.eye(4)
        T_cw_1 = T_21
        uv_0, depth_0 = project_points(self.camera.K, T_cw_0, points_3d)
        uv_1, depth_1 = project_points(self.camera.K, T_cw_1, points_3d)
        error_0 = np.linalg.norm(uv_0 - points_a, axis=1)
        error_1 = np.linalg.norm(uv_1 - points_b, axis=1)

        valid = (
            (depth_0 > config.min_depth)
            & (depth_1 > config.min_depth)
            & (depth_0 < config.max_depth)
            & (depth_1 < config.max_depth)
            & (error_0 < config.max_reproj_error)
            & (error_1 < config.max_reproj_error)
        )
        if int(valid.sum()) < config.min_init_points:
            raise RuntimeError(
                "initialisation failed: too few geometrically valid 3D points "
                f"({int(valid.sum())})"
            )

        points_3d = points_3d[valid]
        points_a = points_a[valid]
        points_b = points_b[valid]
        matches = matches[valid]

        self.poses = [np.eye(4), invert_pose(T_21)]
        self.points = [point.copy() for point in points_3d]
        self._init_point_count = len(self.points)
        self.colors = [
            self._pixel_color(self.images[0], point) for point in points_a
        ]
        self.kp_to_point = {}
        self.observations = []
        for map_index, (match, uv_0, uv_1) in enumerate(
            zip(matches, points_a, points_b)
        ):
            self.kp_to_point[(0, int(match[0]))] = map_index
            self.kp_to_point[(1, int(match[1]))] = map_index
            self.observations.append((0, map_index, uv_0.copy()))
            self.observations.append((1, map_index, uv_1.copy()))

        self.stats["init_matches"] = int(len(matches))
        if self.config.verbose:
            print(
                f"[slam] initialised with {len(matches)} matches, "
                f"{len(self.points)} map points"
            )

    # ------------------------------------------------------------------
    # One keyframe
    # ------------------------------------------------------------------
    def _add_keyframe(self, previous: int, current: int) -> dict[str, int]:
        config = self.config
        features_prev = self.features[previous]
        features_curr = self.features[current]
        stats = {"matches": 0, "inliers": 0, "pnp": 0, "new_points": 0}

        matches = self.matcher.match(
            features_prev,
            features_curr,
            self.images[previous],
            self.images[current],
        )
        stats["matches"] = int(len(matches))
        if len(matches) < config.min_matches:
            self._append_lost_pose()
            return stats

        points_prev = features_prev.points[matches[:, 0]]
        points_curr = features_curr.points[matches[:, 1]]
        F, mask_f = fundamental_matrix(
            points_prev,
            points_curr,
            ransac_threshold=config.f_ransac_threshold,
            confidence=config.f_confidence,
        )
        if F is None or int(mask_f.sum()) < config.min_matches:
            self._append_lost_pose()
            return stats

        points_prev = points_prev[mask_f]
        points_curr = points_curr[mask_f]
        matches = matches[mask_f]
        stats["inliers"] = int(len(matches))

        pnp_object: list[np.ndarray] = []
        pnp_image: list[np.ndarray] = []
        pnp_assoc: list[tuple[int, int, np.ndarray]] = []
        new_matches: list[tuple[int, int, np.ndarray, np.ndarray]] = []

        for match, uv_prev, uv_curr in zip(matches, points_prev, points_curr):
            key = (previous, int(match[0]))
            map_index = self.kp_to_point.get(key)
            if map_index is None:
                new_matches.append(
                    (int(match[0]), int(match[1]), uv_prev, uv_curr)
                )
            else:
                pnp_object.append(self.points[map_index])
                pnp_image.append(uv_curr)
                pnp_assoc.append((map_index, int(match[1]), uv_curr))

        T_cw_curr: np.ndarray | None = None
        pnp_inliers = np.zeros(len(pnp_object), dtype=bool)
        if len(pnp_object) >= config.min_pnp_points:
            T_cw_curr, pnp_inliers = pnp_pose(
                np.asarray(pnp_object),
                np.asarray(pnp_image),
                self.camera.K,
                self.camera.dist,
                ransac=True,
                reprojection_threshold=config.pnp_reprojection_threshold,
            )
            stats["pnp"] = int(len(pnp_object))

        if T_cw_curr is None or int(pnp_inliers.sum()) < config.min_pnp_points:
            # Fall back to the relative pose from the essential matrix.
            E = essential_from_fundamental(F, self.camera.K, self.camera.K)
            T_21, mask_pose = relative_pose_matrix(
                E, points_prev, points_curr, self.camera.K
            )
            if int(mask_pose.sum()) < config.min_matches:
                self._append_lost_pose()
                return stats
            T_cw_curr = T_21
            T_wc_curr = self.poses[previous] @ invert_pose(T_21)
        else:
            T_wc_curr = invert_pose(T_cw_curr)
            inlier_indices = np.flatnonzero(pnp_inliers)
            used_current: set[int] = set()
            for index in inlier_indices:
                map_index, curr_index, uv_curr = pnp_assoc[int(index)]
                if curr_index in used_current:
                    continue
                used_current.add(curr_index)
                self.kp_to_point[(current, curr_index)] = map_index
                self.observations.append(
                    (current, map_index, uv_curr.copy())
                )

        self.poses.append(T_wc_curr)

        # Triangulate matches that do not yet have a 3D point.
        if len(new_matches) >= 4:
            T_cw_prev = invert_pose(self.poses[previous])
            T_cw_curr = invert_pose(T_wc_curr)
            P_prev = self.camera.K @ T_cw_prev[:3, :]
            P_curr = self.camera.K @ T_cw_curr[:3, :]
            prev_uv = np.asarray(
                [item[2] for item in new_matches], dtype=np.float64
            )
            curr_uv = np.asarray(
                [item[3] for item in new_matches], dtype=np.float64
            )

            X_h = cv2.triangulatePoints(
                P_prev, P_curr, prev_uv.T, curr_uv.T
            )
            denominator = X_h[3]
            denominator = np.where(
                np.abs(denominator) < 1e-12, np.nan, denominator
            )
            points_world = (X_h[:3] / denominator).T

            uv_prev_pred, depth_prev = project_points(
                self.camera.K, T_cw_prev, points_world
            )
            uv_curr_pred, depth_curr = project_points(
                self.camera.K, T_cw_curr, points_world
            )
            error_prev = np.linalg.norm(uv_prev_pred - prev_uv, axis=1)
            error_curr = np.linalg.norm(uv_curr_pred - curr_uv, axis=1)
            centre_prev = self.poses[previous][:3, 3]
            centre_curr = T_wc_curr[:3, 3]
            angles = np.asarray(
                [
                    triangulation_angle_deg(centre_prev, centre_curr, point)
                    for point in points_world
                ]
            )
            valid = (
                np.isfinite(points_world).all(axis=1)
                & (depth_prev > config.min_depth)
                & (depth_curr > config.min_depth)
                & (depth_curr < config.max_depth)
                & (error_prev < config.max_reproj_error)
                & (error_curr < config.max_reproj_error)
                & (angles > config.min_triangulation_angle)
            )

            for index in np.flatnonzero(valid):
                map_index = len(self.points)
                prev_index, curr_index, uv_prev, uv_curr = new_matches[int(index)]
                self.points.append(points_world[int(index)].copy())
                self.colors.append(
                    self._pixel_color(self.images[previous], uv_prev)
                )
                self.kp_to_point[(previous, prev_index)] = map_index
                self.kp_to_point[(current, curr_index)] = map_index
                self.observations.append(
                    (previous, map_index, uv_prev.copy())
                )
                self.observations.append(
                    (current, map_index, uv_curr.copy())
                )
                stats["new_points"] += 1

        return stats

    def _append_lost_pose(self) -> None:
        self.poses.append(self.poses[-1].copy())

    # ------------------------------------------------------------------
    # Refinement
    # ------------------------------------------------------------------
    def _refine(self, iterations: int) -> None:
        """Alternating bundle adjustment (block coordinate descent).

        With the 3D points fixed, each camera is re-localised by PnP.  With
        the poses fixed, each point is re-triangulated from all of its
        observations.  A few sweeps reduce drift without a large nonlinear
        solver.  Camera 0 stays fixed to remove the gauge freedom.
        """
        if len(self.poses) < 2 or len(self.points) < 6:
            return
        dist = (
            self.camera.dist
            if self.camera.dist.size
            else np.zeros(5, dtype=np.float64)
        )

        for _ in range(int(iterations)):
            for camera_index in range(1, len(self.poses)):
                observations = [
                    (point_index, uv)
                    for cam, point_index, uv in self.observations
                    if cam == camera_index
                ]
                if len(observations) < 6:
                    continue
                object_points = np.asarray(
                    [self.points[index] for index, _ in observations]
                )
                image_points = np.asarray([uv for _, uv in observations])
                T_cw = invert_pose(self.poses[camera_index])
                rvec, tvec = rvec_tvec_from_pose(T_cw)
                try:
                    ok, rvec, tvec = cv2.solvePnP(
                        object_points,
                        image_points,
                        self.camera.K,
                        dist,
                        rvec,
                        tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                except cv2.error:
                    continue
                if ok:
                    self.poses[camera_index] = invert_pose(
                        pose_from_rvec_tvec(rvec, tvec)
                    )

            observations_by_point: dict[int, list[tuple[int, np.ndarray]]] = (
                defaultdict(list)
            )
            for camera_index, point_index, uv in self.observations:
                observations_by_point[point_index].append(
                    (camera_index, uv)
                )

            for point_index, observations in observations_by_point.items():
                if len(observations) < 2:
                    continue
                point_observations = [
                    (invert_pose(self.poses[cam]), uv)
                    for cam, uv in observations
                ]
                point = triangulate_multiview(
                    point_observations, self.camera.K
                )
                if point is None or not np.all(np.isfinite(point)):
                    continue
                depths = []
                for cam, _ in observations:
                    _, depth = project_points(
                        self.camera.K,
                        invert_pose(self.poses[cam]),
                        point[None, :],
                    )
                    depths.append(depth[0])
                if sum(depth > self.config.min_depth for depth in depths) >= 2:
                    self.points[point_index] = point

    def _rms_reprojection(self) -> float:
        errors = []
        for camera_index, point_index, uv in self.observations:
            uv_pred, depth = project_points(
                self.camera.K,
                invert_pose(self.poses[camera_index]),
                self.points[point_index][None, :],
            )
            if depth[0] > 1e-9:
                errors.append(float(np.linalg.norm(uv_pred[0] - uv)))
        if not errors:
            return float("inf")
        return float(np.sqrt(np.mean(np.square(errors))))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _pixel_color(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        x = int(np.clip(round(float(uv[0])), 0, width - 1))
        y = int(np.clip(round(float(uv[1])), 0, height - 1))
        return np.asarray(image[y, x], dtype=np.float64).reshape(3)
