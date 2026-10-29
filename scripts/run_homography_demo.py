#!/usr/bin/env python3
"""Planar homography and mosaic demo on a synthetic textured wall.

This is the planar counterpart to the epipolar pipeline.  It renders two
views of one textured plane, estimates the homography with RANSAC, and warps
the first view into the second to build a small mosaic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vslam.features import SiftMatcher  # noqa: E402
from vslam.homography import (  # noqa: E402
    crop_black_borders,
    estimate_homography,
    warp_pair,
)
from vslam.simulation import make_planar_scene, render_view  # noqa: E402
from vslam.visualization import draw_matches  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/homography")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-features", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    scene = make_planar_scene(seed=args.seed)
    image0 = render_view(scene, scene.trajectory[0])
    image1 = render_view(scene, scene.trajectory[1])

    matcher = SiftMatcher(max_features=args.max_features, ratio=0.78)
    features0 = matcher.detect(image0)
    features1 = matcher.detect(image1)
    matches = matcher.match(features0, features1, image0, image1)
    if len(matches) < 4:
        raise RuntimeError("not enough matches for a homography")

    points0 = features0.points[matches[:, 0]]
    points1 = features1.points[matches[:, 1]]
    H, mask = estimate_homography(points0, points1)
    if H is None:
        raise RuntimeError("homography estimation failed")

    inliers = int(mask.sum())
    print(f"[homography] matches={len(matches)} inliers={inliers}")
    print(np.array2string(H, precision=4, suppress_small=True))

    draw_matches(
        image0,
        image1,
        points0[mask],
        points1[mask],
        output / "matches.png",
    )
    canvas = crop_black_borders(warp_pair(image0, image1, H))
    cv2.imwrite(str(output / "view0.jpg"), cv2.cvtColor(image0, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "view1.jpg"), cv2.cvtColor(image1, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / "mosaic.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    (output / "homography.txt").write_text(
        np.array2string(H, precision=8, suppress_small=True), encoding="utf-8"
    )
    print(f"[homography] output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
