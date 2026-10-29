#!/usr/bin/env python3
"""End-to-end demo on a synthetic scene with known ground truth.

The script renders a textured 3D scene, selects keyframes, runs the
incremental SLAM pipeline, aligns the estimate to ground truth and writes
plots plus a metrics file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vslam.evaluate import trajectory_metrics  # noqa: E402
from vslam.features import create_matcher  # noqa: E402
from vslam.geometry import fundamental_matrix, invert_pose  # noqa: E402
from vslam.keyframes import select_keyframes  # noqa: E402
from vslam.pipeline import IncrementalSlam, SlamConfig  # noqa: E402
from vslam.simulation import (  # noqa: E402
    generate_dataset,
    load_ground_truth,
    make_scene,
)
from vslam.visualization import (  # noqa: E402
    draw_matches,
    plot_map,
    plot_trajectory,
    save_trajectory_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/synthetic", help="output directory")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--patches", type=int, default=90)
    parser.add_argument("--noise", type=float, default=2.0)
    parser.add_argument("--matcher", default="sift", choices=["sift", "superglue"])
    parser.add_argument("--max-features", type=int, default=2000)
    parser.add_argument("--keyframe-threshold", type=float, default=8.0)
    parser.add_argument("--max-keyframes", type=int, default=10)
    parser.add_argument("--max-depth", type=float, default=12.0)
    parser.add_argument("--max-reproj", type=float, default=1.5)
    parser.add_argument("--min-angle", type=float, default=1.5)
    parser.add_argument("--refine-iterations", type=int, default=3)
    parser.add_argument("--refine-every", type=int, default=2)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument(
        "--reuse-dataset",
        action="store_true",
        help="reuse rendered images if they already exist",
    )
    return parser.parse_args()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    dataset_dir = output / "dataset"
    scene = make_scene(
        seed=args.seed,
        n_patches=args.patches,
        n_frames=args.frames,
        noise_sigma=args.noise,
    )
    expected = dataset_dir / "images" / f"frame_{args.frames - 1:03d}.jpg"
    if not (args.reuse_dataset and expected.is_file()):
        print("[demo] rendering synthetic sequence...")
        paths = generate_dataset(scene, dataset_dir)
    else:
        paths = sorted((dataset_dir / "images").glob("frame_*.jpg"))

    if len(paths) < 3:
        raise RuntimeError("the synthetic sequence needs at least three frames")

    keyframe_indices = select_keyframes(
        paths,
        translation_threshold=args.keyframe_threshold,
        max_keyframes=args.max_keyframes,
    )
    if len(keyframe_indices) < 2:
        keyframe_indices = list(range(len(paths)))

    print(f"[demo] {len(paths)} frames -> {len(keyframe_indices)} keyframes")
    images = [load_rgb(paths[index]) for index in keyframe_indices]

    ground_truth = load_ground_truth(dataset_dir / "ground_truth.csv")
    selected_truth = [ground_truth[index] for index in keyframe_indices]
    # The SLAM world frame is the first keyframe, not the simulator world.
    first_inverse = invert_pose(selected_truth[0])
    relative_truth = [first_inverse @ pose for pose in selected_truth]

    matcher = (
        create_matcher("sift", max_features=args.max_features)
        if args.matcher == "sift"
        else create_matcher("superglue")
    )
    config = SlamConfig(
        max_depth=args.max_depth,
        max_reproj_error=args.max_reproj,
        min_triangulation_angle=args.min_angle,
        refine_iterations=0 if args.no_refine else args.refine_iterations,
        refine_every=args.refine_every,
        verbose=True,
    )

    started = time.perf_counter()
    slam = IncrementalSlam(scene.camera, matcher, config)
    result = slam.run(images, keyframe_indices)
    elapsed = time.perf_counter() - started

    metrics = trajectory_metrics(result.poses, relative_truth)
    frame_names = [paths[index].name for index in keyframe_indices]

    save_trajectory_csv(
        result.poses, frame_names, output / "trajectory_estimated.csv"
    )
    save_trajectory_csv(
        relative_truth, frame_names, output / "trajectory_ground_truth.csv"
    )
    plot_trajectory(
        relative_truth,
        metrics["aligned_xyz"],
        output / "trajectory.png",
        title="Synthetic vSLAM - trajectory (Sim(3) aligned estimate)",
    )
    plot_map(
        result.points,
        result.colors,
        result.poses,
        output / "map.png",
        title=f"Sparse map ({len(result.points)} points)",
    )

    # A match contact sheet makes the feature stage easy to inspect.
    features_a = matcher.detect(images[0])
    features_b = matcher.detect(images[1])
    matches = matcher.match(features_a, features_b, images[0], images[1])
    if len(matches) >= 8:
        points_a = features_a.points[matches[:, 0]]
        points_b = features_b.points[matches[:, 1]]
        _, inlier_mask = fundamental_matrix(
            points_a,
            points_b,
            ransac_threshold=config.f_ransac_threshold,
        )
        draw_matches(
            images[0],
            images[1],
            points_a[inlier_mask],
            points_b[inlier_mask],
            output / "matches.png",
        )

    preview = cv2.imread(str(paths[keyframe_indices[0]]))
    if preview is not None:
        cv2.imwrite(str(output / "scene_preview.jpg"), preview)

    report = {
        "config": asdict(config),
        "keyframe_indices": keyframe_indices,
        "keyframe_names": frame_names,
        "stats": result.stats,
        "elapsed_seconds": elapsed,
        "metrics": {
            key: value
            for key, value in metrics.items()
            if key not in {"aligned_xyz", "rotation", "translation"}
        },
    }
    (output / "metrics.json").write_text(
        json.dumps(_jsonable(report), indent=2), encoding="utf-8"
    )

    print("\n[demo] results")
    print(f"  keyframes        : {len(keyframe_indices)}")
    print(f"  map points       : {len(result.points)}")
    print(f"  ATE RMSE         : {metrics['ate_rmse']:.4f}")
    print(f"  rotation RPE mean: {metrics['rpe_rotation_mean_deg']:.3f} deg")
    print(f"  translation RPE  : {metrics['rpe_translation_mean']:.4f}")
    print(f"  runtime          : {elapsed:.1f} s")
    print(f"  output           : {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
