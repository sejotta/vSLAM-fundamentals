"""A small, self-contained synthetic scene generator.

The simulator avoids any external robotics stack.  A scene is a cloud of
textured planar patches in 3D; a camera follows a smooth path and each patch
is rendered with a perspective warp.  The result is a sequence of images with
exact ground-truth camera poses, which makes the SLAM pipeline reproducible
and gives us something to measure against.

The renderer uses the painter's algorithm (far patches first) and a pinhole
camera model.  It is not a physically based renderer, but it produces
realistic parallax, scale changes and perspective distortion, which is what
the geometry code needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .camera import Camera


@dataclass
class TexturedPatch:
    """A textured rectangle in 3D.

    ``u`` and ``v`` are half-axis vectors.  The four corners are
    ``centre - u - v``, ``centre + u - v``, ``centre + u + v`` and
    ``centre - u + v``, matching the texture corners
    ``(0, 0)``, ``(w, 0)``, ``(w, h)`` and ``(0, h)``.
    """

    centre: np.ndarray
    u: np.ndarray
    v: np.ndarray
    texture: np.ndarray

    @property
    def corners(self) -> np.ndarray:
        c, u, v = self.centre, self.u, self.v
        return np.stack([c - u - v, c + u - v, c + u + v, c - u + v])

    @property
    def normal(self) -> np.ndarray:
        n = np.cross(self.u, self.v)
        norm = np.linalg.norm(n)
        return n / norm if norm > 0 else n


@dataclass
class SyntheticScene:
    """Camera, patches and ground-truth trajectory."""

    camera: Camera
    patches: list[TexturedPatch]
    trajectory: list[np.ndarray]
    noise_sigma: float = 2.0
    seed: int = 0
    background: tuple[int, int, int] = (8, 8, 12)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False
    )


def _random_texture(
    rng: np.random.Generator, width: int = 64, height: int = 64
) -> np.ndarray:
    """Generate a muted, high-contrast texture with many stable corners.

    The pattern is created in grayscale (which is what SIFT consumes) and
    then tinted with a single random colour.  This looks more like a painted
    panel than rainbow noise and produces cleaner feature descriptors.
    """
    grid_w = max(3, width // 12)
    grid_h = max(3, height // 12)
    low_frequency = rng.integers(
        35, 225, size=(grid_h, grid_w), dtype=np.uint8
    )
    texture = cv2.resize(
        low_frequency, (width, height), interpolation=cv2.INTER_CUBIC
    )

    n_shapes = max(14, min(60, (width * height) // 4000))
    max_rect_w = max(8, width // 6)
    max_rect_h = max(8, height // 6)
    max_radius = max(4, min(width, height) // 8)
    for _ in range(n_shapes):
        value = int(rng.integers(25, 240))
        x = int(rng.integers(0, max(1, width - 8)))
        y = int(rng.integers(0, max(1, height - 8)))
        if rng.random() < 0.55:
            w = int(rng.integers(6, max_rect_w))
            h = int(rng.integers(6, max_rect_h))
            cv2.rectangle(
                texture,
                (x, y),
                (min(x + w, width - 1), min(y + h, height - 1)),
                value,
                -1,
            )
        else:
            radius = int(rng.integers(3, max_radius))
            cv2.circle(texture, (x, y), radius, value, -1)

    n_lines = max(5, min(30, (width * height) // 8000))
    for _ in range(n_lines):
        start = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        end = (
            int(rng.integers(0, width)),
            int(rng.integers(0, height)),
        )
        cv2.line(
            texture,
            start,
            end,
            int(rng.integers(0, 255)),
            int(rng.integers(1, 3)),
        )

    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    tint = rng.uniform(0.55, 1.0, size=3)
    colour_texture = np.clip(
        texture.astype(np.float64)[:, :, None] * tint[None, None, :],
        0,
        255,
    )
    noise = rng.normal(0.0, 4.0, colour_texture.shape)
    return np.clip(colour_texture + noise, 0, 255).astype(np.uint8)


def look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: Sequence[float] = (0.0, -1.0, 0.0),
) -> np.ndarray:
    """Build ``T_wc`` for a camera at ``eye`` looking at ``target``.

    The camera convention is OpenCV's: ``x`` right, ``y`` down, ``z``
    forward.  ``up`` therefore defaults to ``(0, -1, 0)`` in a world whose
    ``y`` axis points down.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)

    up_vec = np.asarray(up, dtype=np.float64)
    up_vec /= np.linalg.norm(up_vec)
    down = -up_vec

    right = np.cross(down, forward)
    norm = np.linalg.norm(right)
    if norm < 1e-9:
        # Camera looking straight up/down: pick an arbitrary right vector.
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= norm
    down_camera = np.cross(forward, right)

    T_wc = np.eye(4)
    T_wc[:3, :3] = np.column_stack([right, down_camera, forward])
    T_wc[:3, 3] = eye
    return T_wc


def _make_camera_path(
    n_frames: int,
    target: Sequence[float],
    radius: float,
    half_angle_deg: float,
) -> list[np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    angles = np.deg2rad(
        np.linspace(-half_angle_deg, half_angle_deg, n_frames)
    )
    poses: list[np.ndarray] = []
    for index, angle in enumerate(angles):
        t = index / max(1, n_frames - 1)
        local_radius = radius + 0.35 * np.sin(2 * np.pi * t)
        eye = target + np.array(
            [
                local_radius * np.sin(angle),
                0.18 * np.sin(2 * np.pi * t),
                -local_radius * np.cos(angle),
            ]
        )
        poses.append(look_at(eye, target))
    return poses


def make_scene(
    seed: int = 7,
    n_patches: int = 90,
    n_frames: int = 24,
    width: int = 640,
    height: int = 480,
    focal: float = 650.0,
    noise_sigma: float = 2.0,
    half_angle_deg: float = 26.0,
    radius: float = 5.4,
) -> SyntheticScene:
    """Create a reproducible floating-patch scene and camera path."""
    rng = np.random.default_rng(seed)
    camera = Camera(
        fx=focal,
        fy=focal,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        width=width,
        height=height,
    )

    target = np.array([0.0, 0.0, 4.4])
    patches: list[TexturedPatch] = []
    for _ in range(n_patches):
        centre = np.array(
            [
                rng.uniform(-3.1, 3.1),
                rng.uniform(-2.0, 2.0),
                rng.uniform(3.1, 7.0),
            ]
        )
        size = float(rng.uniform(0.55, 1.05))
        yaw = float(rng.uniform(-0.45, 0.45))
        pitch = float(rng.uniform(-0.22, 0.22))

        u = np.array([np.cos(yaw), 0.0, np.sin(yaw)]) * (size / 2.0)
        v = np.array([0.0, np.cos(pitch), -np.sin(pitch)]) * (size / 2.0)
        texture = _random_texture(rng, width=72, height=72)
        patches.append(
            TexturedPatch(centre=centre, u=u, v=v, texture=texture)
        )

    trajectory = _make_camera_path(n_frames, target, radius, half_angle_deg)
    return SyntheticScene(
        camera=camera,
        patches=patches,
        trajectory=trajectory,
        noise_sigma=noise_sigma,
        seed=seed,
        _rng=np.random.default_rng(seed + 1),
    )


def make_planar_scene(
    seed: int = 11,
    width: int = 640,
    height: int = 480,
    focal: float = 650.0,
) -> SyntheticScene:
    """Create a single large textured wall for a homography demo."""
    rng = np.random.default_rng(seed)
    camera = Camera(
        fx=focal,
        fy=focal,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        width=width,
        height=height,
    )
    wall = TexturedPatch(
        centre=np.array([0.0, 0.0, 6.0]),
        u=np.array([4.2, 0.0, 0.0]),
        v=np.array([0.0, 3.2, 0.0]),
        texture=_random_texture(rng, width=512, height=384),
    )

    pose0 = np.eye(4)
    pose1 = look_at(
        np.array([0.55, 0.18, 0.0]),
        np.array([0.45, 0.12, 6.0]),
    )
    # A small roll makes the planar alignment less trivial.
    roll = np.deg2rad(2.0)
    R_roll = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0.0],
            [np.sin(roll), np.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pose1 = pose1.copy()
    pose1[:3, :3] = pose1[:3, :3] @ R_roll

    return SyntheticScene(
        camera=camera,
        patches=[wall],
        trajectory=[pose0, pose1],
        noise_sigma=1.0,
        seed=seed,
        background=(20, 20, 24),
        _rng=np.random.default_rng(seed + 1),
    )


def render_view(
    scene: SyntheticScene,
    T_wc: np.ndarray,
    noise_sigma: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render one RGB image from a camera pose ``T_wc`` (camera -> world)."""
    camera = scene.camera
    width, height = camera.width, camera.height
    K = camera.K
    T_wc = np.asarray(T_wc, dtype=np.float64)
    centre = T_wc[:3, 3]
    T_cw = np.eye(4)
    T_cw[:3, :3] = T_wc[:3, :3].T
    T_cw[:3, 3] = -T_wc[:3, :3].T @ centre
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]

    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = scene.background

    order = sorted(
        range(len(scene.patches)),
        key=lambda i: -float(
            np.linalg.norm(scene.patches[i].centre - centre)
        ),
    )

    for patch_index in order:
        patch = scene.patches[patch_index]
        corners_cam = (R @ patch.corners.T + t[:, None]).T
        if np.any(corners_cam[:, 2] <= 0.05):
            continue

        homogeneous = (K @ corners_cam.T).T
        uv = homogeneous[:, :2] / homogeneous[:, 2:3]
        if (
            uv[:, 0].max() < 0
            or uv[:, 0].min() > width
            or uv[:, 1].max() < 0
            or uv[:, 1].min() > height
        ):
            continue

        texture_h, texture_w = patch.texture.shape[:2]
        source = np.float32(
            [[0, 0], [texture_w, 0], [texture_w, texture_h], [0, texture_h]]
        )
        destination = uv.astype(np.float32)
        transform = cv2.getPerspectiveTransform(source, destination)

        warped = cv2.warpPerspective(
            patch.texture,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        mask = cv2.warpPerspective(
            np.full((texture_h, texture_w), 255, dtype=np.uint8),
            transform,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        canvas[mask > 0] = warped[mask > 0]

    sigma = scene.noise_sigma if noise_sigma is None else noise_sigma
    if sigma and sigma > 0:
        generator = rng or scene._rng
        noisy = canvas.astype(np.float64) + generator.normal(
            0.0, sigma, canvas.shape
        )
        canvas = np.clip(noisy, 0, 255).astype(np.uint8)
    return canvas


def generate_dataset(
    scene: SyntheticScene,
    output_dir: str | Path,
    prefix: str = "frame",
    jpeg_quality: int = 95,
) -> list[Path]:
    """Render a scene to ``output_dir/images`` and write ground truth.

    The ground-truth CSV stores the camera centre and the nine rotation
    entries for every frame.  ``camera.json`` stores the intrinsics.
    """
    output = Path(output_dir)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(scene.seed + 100)
    paths: list[Path] = []
    for index, pose in enumerate(scene.trajectory):
        image = render_view(scene, pose, rng=rng)
        path = image_dir / f"{prefix}_{index:03d}.jpg"
        cv2.imwrite(
            str(path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )
        paths.append(path)

    camera = scene.camera
    camera_data = {
        "fx": camera.fx,
        "fy": camera.fy,
        "cx": camera.cx,
        "cy": camera.cy,
        "width": camera.width,
        "height": camera.height,
        "dist": camera.dist.tolist(),
    }
    (output / "camera.json").write_text(
        json.dumps(camera_data, indent=2), encoding="utf-8"
    )

    with (output / "ground_truth.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        import csv

        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame", "x", "y", "z",
                "r00", "r01", "r02",
                "r10", "r11", "r12",
                "r20", "r21", "r22",
            ]
        )
        for index, pose in enumerate(scene.trajectory):
            writer.writerow(
                [
                    f"{prefix}_{index:03d}.jpg",
                    *pose[:3, 3].tolist(),
                    *pose[:3, :3].ravel().tolist(),
                ]
            )

    return paths


def load_ground_truth(
    csv_path: str | Path,
) -> list[np.ndarray]:
    """Load poses written by :func:`generate_dataset`."""
    import csv

    poses: list[np.ndarray] = []
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            T = np.eye(4)
            T[:3, :3] = np.array(
                [
                    [float(row["r00"]), float(row["r01"]), float(row["r02"])],
                    [float(row["r10"]), float(row["r11"]), float(row["r12"])],
                    [float(row["r20"]), float(row["r21"]), float(row["r22"])],
                ]
            )
            T[:3, 3] = [float(row["x"]), float(row["y"]), float(row["z"])]
            poses.append(T)
    return poses
