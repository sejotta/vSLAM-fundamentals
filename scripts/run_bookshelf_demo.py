#!/usr/bin/env python3
"""Reconstruct the bookshelf example with incremental PnP + triangulation.

This is the real-data demo: five calibrated keyframes from a handheld pan
across a bookshelf.  Every new keyframe is localised by PnP against the 3D
points already in the map; matches without a 3D point are triangulated and
added to the map.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vslam.camera import Camera  # noqa: E402
from vslam.features import create_matcher  # noqa: E402
from vslam.geometry import fundamental_matrix  # noqa: E402
from vslam.pipeline import IncrementalSlam, SlamConfig  # noqa: E402
from vslam.visualization import (  # noqa: E402
    draw_matches,
    plot_map,
    plot_trajectory,
    save_trajectory_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/bookshelf")
    parser.add_argument("--matcher", default="sift", choices=["sift", "superglue"])
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--refine-iterations", type=int, default=3)
    parser.add_argument("--max-depth", type=float, default=50.0)
    parser.add_argument("--max-reproj", type=float, default=2.5)
    parser.add_argument("--no-refine", action="store_true")
    return parser.parse_args()


def load_camera(path: Path) -> tuple[Camera, Camera]:
    data = json.loads(path.read_text(encoding="utf-8"))
    kw = dict(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        width=int(data["width"]),
        height=int(data["height"]),
    )
    calibration = Camera(dist=np.asarray(data.get("dist", []), dtype=float), **kw)
    geometry = Camera(dist=np.zeros(5), **kw)
    return calibration, geometry


def make_montage(frames: list[np.ndarray], names: list[str]) -> np.ndarray:
    tiles = []
    for image, name in zip(frames, names):
        thumb = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
        tile = np.zeros((270, 320, 3), dtype=np.uint8)
        tile[30:] = thumb
        cv2.putText(
            tile,
            name,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    return cv2.hconcat(tiles)


def main() -> int:
    args = parse_args()
    example = ROOT / "examples" / "bookshelf"
    image_paths = sorted((example / "images").glob("*.jpg"))
    if len(image_paths) < 3:
        raise RuntimeError("bookshelf example needs at least three frames")

    calibration, camera = load_camera(example / "camera.json")
    images_bgr = [cv2.imread(str(path)) for path in image_paths]
    if any(image is None for image in images_bgr):
        raise FileNotFoundError("could not read one of the bookshelf frames")

    images_bgr = [
        cv2.undistort(image, calibration.K, calibration.dist)
        if calibration.dist.size and np.any(calibration.dist)
        else image
        for image in images_bgr
    ]
    images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images_bgr]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    matcher = (
        create_matcher("sift", max_features=args.max_features, ratio=0.75)
        if args.matcher == "sift"
        else create_matcher("superglue")
    )
    config = SlamConfig(
        max_depth=args.max_depth,
        max_reproj_error=args.max_reproj,
        min_triangulation_angle=0.5,
        refine_iterations=0 if args.no_refine else args.refine_iterations,
        refine_every=2,
        verbose=True,
    )

    started = time.perf_counter()
    keyframes = list(range(len(images)))
    result = IncrementalSlam(camera, matcher, config).run(images, keyframes)
    elapsed = time.perf_counter() - started

    names = [path.name for path in image_paths]
    save_trajectory_csv(result.poses, names, output / "trajectory_estimated.csv")
    plot_map(
        result.points,
        result.colors,
        result.poses,
        output / "map.png",
        title=f"Bookshelf reconstruction ({len(result.points)} points)",
    )
    plot_trajectory(
        None,
        result.poses,
        output / "trajectory.png",
        title="Bookshelf sequence - PnP camera trajectory",
    )

    features_a = matcher.detect(images[0])
    features_b = matcher.detect(images[1])
    matches = matcher.match(features_a, features_b, images[0], images[1])
    if len(matches) >= 8:
        points_a = features_a.points[matches[:, 0]]
        points_b = features_b.points[matches[:, 1]]
        _, inlier_mask = fundamental_matrix(points_a, points_b)
        draw_matches(
            images[0],
            images[1],
            points_a[inlier_mask],
            points_b[inlier_mask],
            output / "matches.png",
            max_matches=60,
        )

    cv2.imwrite(str(output / "frame_000.jpg"), images_bgr[0])
    cv2.imwrite(
        str(output / "input_montage.jpg"),
        make_montage(images_bgr, names),
    )
    try:
        from PIL import Image

        pil_frames = [
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            for image in images_bgr
        ]
        pil_frames[0].save(
            output / "input_sequence.gif",
            save_all=True,
            append_images=pil_frames[1:],
            duration=700,
            loop=0,
            optimize=True,
        )
    except ImportError:
        pass

    report = {
        "frames": names,
        "stats": result.stats,
        "elapsed_seconds": elapsed,
        "camera": {
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("\n[bookshelf] reconstruction")
    print(f"  keyframes       : {len(images)}")
    print(f"  map points      : {len(result.points)}")
    print(f"  RMS reprojection: {result.stats['rms_reprojection_px']:.2f} px")
    print(f"  runtime         : {elapsed:.2f} s")
    print(f"  output          : {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
