# Context and curation notes

This repository is a public, cleaned-up cut of two private Python prototypes
that implemented an **incremental 3D reconstruction with PnP**. The goal was
to keep the parts that show engineering and computer-vision work — algorithms,
structure, documentation and real results — while leaving out everything that
is personal, institutional, proprietary or simply noise.

---

## What was kept

| Area | Public implementation |
|---|---|
| Feature detection and matching | `vslam/features.py` (SIFT + ratio test, optional SuperPoint/SuperGlue) |
| Two-view geometry | `vslam/geometry.py` (F, E, `recoverPose`, triangulation, PnP) |
| Planar geometry and mosaicking | `vslam/homography.py` |
| Keyframe selection | `vslam/keyframes.py` (optical flow) |
| Incremental reconstruction | `vslam/pipeline.py` (PnP localisation + new triangulation) |
| Pose/point refinement | `vslam/pipeline.py` (alternating bundle adjustment) |
| 3D visualisation | `vslam/visualization.py` |
| Trajectory evaluation | `vslam/evaluate.py` (Umeyama Sim(3), ATE, RPE) |
| Real example | `examples/bookshelf/` (five frames + calibration) |
| Real-data demo | `scripts/run_bookshelf_demo.py` |
| Generic sequence runner | `scripts/run_sequence.py` |
| Synthetic regression harness | `vslam/simulation.py`, `scripts/run_synthetic_slam.py` |
| Tests | `tests/test_geometry.py` |

The code was restructured rather than copied file-for-file: hardcoded paths
were removed, duplicate experiments were merged, coordinate conventions were
documented, and the pipeline was split into modules with a stable interface.

---

## What was deliberately left out

| Excluded material | Reason |
|---|---|
| Student/author names, emails, IDs, grades, personal comments | Privacy; irrelevant to the technical work |
| University, course, practice and report identifiers | The portfolio should stand on its own, not read as coursework |
| PDFs, statements, presentations, exam files, LaTeX sources | Course logistics, not code or explanation |
| Absolute Windows paths and local folder names | Not portable, and they leak the local environment |
| Most of the private image sequence and generated media | Only five non-identifying keyframes are included as a reproducibility example; the rest is not redistributed |
| SuperGluePretrainedNetwork checkout and `.pth` weights | Third-party code/weights with their own licence; kept as an optional external dependency |
| Robotics-simulator scene, robot model, simulator configuration and bridge scripts | The public repository is simulator- and robot-agnostic; no proprietary simulator asset is needed to understand or run the algorithms |
| Old/duplicate experimental scripts and hardcoded-frame variants | Superseded by the cleaned modules and CLI demos |
| Large generated outputs (`runs/`, point clouds, videos) | Reproducible from code; only small documentation images are committed |

No robot model, simulator project, course document, full dataset or personal
identifier is required to build, run or understand this repository.

---

## Source-to-public mapping

The private prototypes contained several near-duplicate scripts. Their useful
parts were consolidated as follows:

| Original role | Public module |
|---|---|
| SuperPoint/SuperGlue wrapper and match conversion | `vslam/features.py` |
| Camera matrix, F/E utilities, PnP, triangulation | `vslam/geometry.py` |
| Two-view reconstruction + PnP trajectory | `vslam/pipeline.py` (`IncrementalSlam`) |
| Incremental keyframe triangulation and pose accumulation | `vslam/pipeline.py` |
| Bundle-adjustment experiment | `vslam/pipeline.py` (`_refine`) |
| Plotly 3D camera/point drawing | `vslam/visualization.py` (Matplotlib port) |
| Homography and cumulative mosaic | `vslam/homography.py` |
| Optical-flow keyframe selection | `vslam/keyframes.py` |
| Fixed-frame experiment scripts | Consolidated into `scripts/run_sequence.py` and `scripts/run_bookshelf_demo.py` |
| Handheld bookshelf sequence | Five keyframes published under `examples/bookshelf/`; the full sequence is not |
| Synthetic renderer used for testing | Kept as `vslam/simulation.py`; it is **not** the project data |

The rewrite also fixed a few issues in the prototypes:

- camera-to-world vs. world-to-camera conventions are now explicit,
- all paths are relative or supplied on the command line,
- images are undistorted before the geometry stage,
- the PnP rotation uses `cv2.Rodrigues` instead of a generic matrix
  exponential,
- the evaluation aligns the estimate to ground truth before computing ATE,
- the bundle-adjustment pass checks cheirality before accepting a refined
  point.

---

## Revision: real-data showcase

The first version of this repository used only synthetic images for the
README. That was not representative of the work: the reconstruction was built
from a real handheld bookshelf sequence using **incremental triangulation with
PnP**. The synthetic images have therefore been removed from the showcase and
replaced by:

- the five actual input frames (`examples/bookshelf/`),
- real SIFT matches between the first two frames,
- the 3D map reconstructed from those frames,
- the estimated PnP camera trajectory,
- the planar mosaic obtained on the same sequence.

The synthetic renderer remains in the repository only as a deterministic
regression harness with exact ground truth. It is clearly labelled as a test,
not as the project's data.

---

## What is intentionally *not* claimed

This is a compact portfolio implementation, not a research or production SLAM
system. It does not implement loop closure, relocalisation, global pose-graph
optimisation, IMU fusion, dense mapping, or a full sparse Levenberg–Marquardt
bundle adjuster. The bookshelf sequence is close to planar and the camera
motion is mostly a pan, which is a difficult case for epipolar geometry; the
result is a qualitative reconstruction, not a metric evaluation. The synthetic
results are a correctness check, not a benchmark against the state of the art.

---

## Verification performed

```bash
python -m unittest discover -s tests -v
python scripts/run_bookshelf_demo.py
python scripts/run_synthetic_slam.py
python scripts/run_homography_demo.py
```

- All five geometry tests pass: triangulation, F/E pose recovery, PnP,
  Umeyama alignment and homography.
- The real bookshelf demo reconstructs **1,600 map points from 5 keyframes**
  with **0.38 px RMS reprojection error** in about **1.8 s** on CPU.
- The synthetic regression scene reports ATE RMSE `0.0096`, rotation RPE
  `0.058°` and RMS reprojection error `0.29 px` for 9 keyframes and 2,555 map
  points. It exists to catch geometry regressions.
- The homography demo estimates a planar transform from 839 matches with 815
  RANSAC inliers and writes a mosaic.

The bookshelf numbers are a qualitative reconstruction: there is no
ground-truth trajectory for that sequence. The synthetic numbers can be
reproduced with the commands above.

---

## If you extend the repository

- Put new datasets under `data/` or `runs/` and keep them out of Git.
- Never hardcode absolute paths; expose them as CLI arguments.
- Keep third-party models and weights outside the repository.
- Add a unit test for every new geometric primitive.
- If you add a real sequence, document its licence and do not include
  personally identifying imagery.
