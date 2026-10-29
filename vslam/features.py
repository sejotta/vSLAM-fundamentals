"""Feature detection and matching back-ends.

Two back-ends are provided:

``SiftMatcher``
    Classical SIFT keypoints with a Lowe ratio test.  It only needs
    OpenCV and is the default, so every example runs on a CPU-only machine.

``SuperPointSuperGlueMatcher``
    The learned SuperPoint detector and SuperGlue matcher.  This back-end is
    optional: it needs PyTorch and a local checkout of the public
    ``SuperGluePretrainedNetwork`` repository.  See the README for setup.

Both back-ends return the same :class:`Features` object and a simple
``(M, 2)`` integer array of matches, which keeps the rest of the pipeline
independent of the implementation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np


@dataclass
class Features:
    """Keypoints and descriptors extracted from one image.

    Attributes
    ----------
    points:
        ``(N, 2)`` pixel coordinates.
    descriptors:
        ``(N, D)`` descriptors.  ``D`` depends on the back-end.
    scores:
        Optional ``(N,)`` detector confidence.
    tensor:
        Optional image tensor kept for learned matchers that need the image
        itself (SuperGlue consumes both descriptors and image context).
    """

    points: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray | None = None
    tensor: Any | None = None


class Matcher(Protocol):
    """Common interface implemented by every feature back-end."""

    name: str

    def detect(self, image_rgb: np.ndarray) -> Features:
        """Detect features in an RGB image."""

    def match(
        self,
        features_a: Features,
        features_b: Features,
        image_a: np.ndarray | None = None,
        image_b: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return an ``(M, 2)`` array of ``(index_a, index_b)`` matches."""


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _empty_features() -> Features:
    return Features(
        points=np.empty((0, 2), dtype=np.float32),
        descriptors=np.empty((0, 0), dtype=np.float32),
        scores=np.empty((0,), dtype=np.float32),
    )


class SiftMatcher:
    """SIFT + mutual ratio-test matching."""

    name = "sift"

    def __init__(
        self,
        max_features: int = 2000,
        ratio: float = 0.75,
        contrast_threshold: float = 0.01,
    ) -> None:
        self.ratio = float(ratio)
        self.sift = cv2.SIFT_create(
            nfeatures=int(max_features),
            contrastThreshold=float(contrast_threshold),
            edgeThreshold=10,
            sigma=1.6,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def detect(self, image_rgb: np.ndarray) -> Features:
        gray = _to_gray(image_rgb)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if not keypoints or descriptors is None:
            return _empty_features()

        points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        scores = np.array([kp.response for kp in keypoints], dtype=np.float32)
        return Features(
            points=points,
            descriptors=np.asarray(descriptors, dtype=np.float32),
            scores=scores,
        )

    def match(
        self,
        features_a: Features,
        features_b: Features,
        image_a: np.ndarray | None = None,
        image_b: np.ndarray | None = None,
    ) -> np.ndarray:
        del image_a, image_b  # SIFT matching does not need the images
        if len(features_a.descriptors) < 2 or len(features_b.descriptors) < 2:
            return np.empty((0, 2), dtype=np.int32)

        knn = self.matcher.knnMatch(
            features_a.descriptors, features_b.descriptors, k=2
        )
        matches: list[tuple[int, int, float]] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            first, second = pair
            if first.distance < self.ratio * second.distance:
                matches.append((first.queryIdx, first.trainIdx, first.distance))

        matches.sort(key=lambda item: item[2])
        return np.array(
            [(i, j) for i, j, _ in matches], dtype=np.int32
        ).reshape(-1, 2)


class SuperPointSuperGlueMatcher:
    """Optional learned detector/matcher.

    The implementation expects the public repository layout::

        third_party/SuperGluePretrainedNetwork/models/superpoint.py

    Set ``SUPERGLUE_REPO`` or pass ``repo_path`` if it lives elsewhere.
    The weights are downloaded by that project, not by this repository.
    """

    name = "superpoint-superglue"

    def __init__(
        self,
        repo_path: str | Path | None = None,
        weights: str = "indoor",
        max_keypoints: int = 1024,
        keypoint_threshold: float = 0.005,
        nms_radius: int = 3,
        sinkhorn_iterations: int = 20,
        match_threshold: float = 0.2,
        device: str | None = None,
    ) -> None:
        try:
            import torch
        except (ImportError, OSError) as exc:  # pragma: no cover - machine dependent
            raise ImportError(
                "PyTorch is required for the SuperPoint/SuperGlue back-end. "
                "Install it with `pip install torch`."
            ) from exc

        repo = Path(
            repo_path
            or os.environ.get("SUPERGLUE_REPO", "")
            or Path(__file__).resolve().parents[1]
            / "third_party"
            / "SuperGluePretrainedNetwork"
        ).expanduser()
        models_dir = repo / "models"
        if not (models_dir / "superpoint.py").is_file():
            raise FileNotFoundError(
                "SuperGluePretrainedNetwork was not found. Clone it with:\n"
                "  git clone https://github.com/magicleap/"
                "SuperGluePretrainedNetwork third_party/SuperGluePretrainedNetwork\n"
                f"Looked in: {repo}"
            )

        repo_str = str(repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from models.superglue import SuperGlue  # type: ignore
        from models.superpoint import SuperPoint  # type: ignore

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.superpoint = SuperPoint(
            {
                "nms_radius": int(nms_radius),
                "keypoint_threshold": float(keypoint_threshold),
                "max_keypoints": int(max_keypoints),
            }
        ).to(self.device).eval()
        self.superglue = SuperGlue(
            {
                "weights": weights,
                "sinkhorn_iterations": int(sinkhorn_iterations),
                "match_threshold": float(match_threshold),
            }
        ).to(self.device).eval()

    def detect(self, image_rgb: np.ndarray) -> Features:
        gray = _to_gray(image_rgb)
        # SuperPoint's encoder requires spatial dimensions divisible by 8.
        pad_h = (8 - gray.shape[0] % 8) % 8
        pad_w = (8 - gray.shape[1] % 8) % 8
        if pad_h or pad_w:
            gray = cv2.copyMakeBorder(
                gray, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0
            )

        tensor = (
            self.torch.from_numpy(gray / 255.0)
            .float()[None, None]
            .to(self.device)
        )
        with self.torch.no_grad():
            pred = self.superpoint({"image": tensor})

        points = pred["keypoints"][0].cpu().numpy().astype(np.float32)
        descriptors = (
            pred["descriptors"][0].cpu().numpy().T.astype(np.float32)
        )
        scores = pred["scores"][0].cpu().numpy().astype(np.float32)
        return Features(
            points=points,
            descriptors=descriptors,
            scores=scores,
            tensor=tensor,
        )

    def match(
        self,
        features_a: Features,
        features_b: Features,
        image_a: np.ndarray | None = None,
        image_b: np.ndarray | None = None,
    ) -> np.ndarray:
        del image_a, image_b
        if features_a.tensor is None or features_b.tensor is None:
            raise ValueError("SuperGlue matching requires features.detect() tensors")
        if len(features_a.points) == 0 or len(features_b.points) == 0:
            return np.empty((0, 2), dtype=np.int32)

        torch = self.torch
        data = {
            "keypoints0": torch.from_numpy(features_a.points)[None].to(self.device),
            "descriptors0": torch.from_numpy(features_a.descriptors.T)[None].to(
                self.device
            ),
            "scores0": torch.from_numpy(features_a.scores)[None].to(self.device),
            "image0": features_a.tensor,
            "keypoints1": torch.from_numpy(features_b.points)[None].to(self.device),
            "descriptors1": torch.from_numpy(features_b.descriptors.T)[None].to(
                self.device
            ),
            "scores1": torch.from_numpy(features_b.scores)[None].to(self.device),
            "image1": features_b.tensor,
        }
        with torch.no_grad():
            pred = self.superglue(data)

        matches0 = pred["matches0"][0].cpu().numpy()
        pairs = [
            (i, int(j))
            for i, j in enumerate(matches0)
            if int(j) >= 0
        ]
        return np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def create_matcher(name: str, **kwargs: Any) -> Matcher:
    """Create a feature back-end by name.

    Parameters
    ----------
    name:
        ``"sift"`` (default) or ``"superglue"``/``"superpoint-superglue"``.
    kwargs:
        Forwarded to the selected class.
    """
    key = name.strip().lower()
    if key == "sift":
        return SiftMatcher(**kwargs)
    if key in {"superglue", "superpoint-superglue", "sp-sg"}:
        return SuperPointSuperGlueMatcher(**kwargs)
    raise ValueError(f"Unknown matcher {name!r}; use 'sift' or 'superglue'")
