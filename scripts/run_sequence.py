#!/usr/bin/env python3
"""Run the SLAM pipeline on an arbitrary folder of images.

The sequence is assumed to come from one calibrated camera and to be ordered
by filename.  Keyframes are selected automatically.  No ground truth is
required; the output is a trajectory CSV and a sparse 3D map.
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

from vslam.camera import Camera  # noqa: E402
from vslam.features import create_matcher  # noqa: E402
from vslam.keyframes import select_keyframes  # noqa: E402
from vslam.pipeline import IncrementalSlam, SlamConfig  # noqa: E402
from vslam.visualization import (  # noqa: E402
    plot_map,
    plot_trajectory,
    save_trajectory_csv,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="folder with the sequence")
    parser.add_argument("--output", default="runs/sequence")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument(
        "--dist",
        type=float,
        nargs="*",
        default=None,
        help="distortion coefficients (k1 k2 p1 p2 k3 ...)",
    )
    parser.add_argument("--matcher", default="sift", choices=["sift", "superglue"])
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--keyframe-threshold", type=float, default=25.0)
    parser.add_argument("--max-keyframes", type=int, default=10)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--max-reproj", type=float, default=2.5)
    parser.add_argument("--refine-iterations", type=int, default=3)
    parser.add_argument("--no-refine", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = Path(args.images)
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) < 3:
        raise RuntimeError("at least three images are required")

    probe = cv2.imread(str(paths[0]))
    if probe is None:
        raise FileNotFoundError(paths[0])
    height, width = probe.shape[:2]

    camera = Camera(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=width,
        height=height,
        dist=np.asarray(args.dist if args.dist else [0, 0, 0, 0, 0], dtype=float),
    )

    keyframes = select_keyframes(
        paths,
        translation_threshold=args.keyframe_threshold,
        max_keyframes=args.max_keyframes,
    )
    if len(keyframes) < 2:
        keyframes = list(range(len(paths)))
    print(f"[sequence] {len(paths)} images -> {len(keyframes)} keyframes")

    images = []
    for index in keyframes:
        image = cv2.imread(str(paths[index]))
        if image is None:
            raise FileNotFoundError(paths[index])
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    matcher = (
        create_matcher("sift", max_features=args.max_features)
        if args.matcher == "sift"
        else create_matcher("superglue")
    )
    config = SlamConfig(
        max_depth=args.max_depth,
        max_reproj_error=args.max_reproj,
        refine_iterations=0 if args.no_refine else args.refine_iterations,
        verbose=True,
    )
    result = IncrementalSlam(camera, matcher, config).run(images, keyframes)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    names = [paths[index].name for index in keyframes]
    save_trajectory_csv(result.poses, names, output / "trajectory_estimated.csv")
    plot_trajectory(
        None,
        result.poses,
        output / "trajectory.png",
        title="Estimated camera trajectory",
    )
    plot_map(
        result.points,
        result.colors,
        result.poses,
        output / "map.png",
        title=f"Sparse map ({len(result.points)} points)",
    )
    print(f"[sequence] output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
