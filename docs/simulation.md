# Synthetic regression harness

This is **not** the project's input data and it is not the main demo. The main
real-data result uses the five bookshelf frames in `examples/bookshelf/`. The
synthetic scene is a deterministic test harness: it provides exact camera
poses so the geometry implementation can be checked automatically, without a
robotics simulator, a robot model, or any external asset.

---

## 1. Why a synthetic harness?

A reconstruction pipeline is easier to test when the ground truth is known:

- every camera pose is exact,
- every 3D point is known,
- the intrinsics are known,
- the amount of noise is controlled,
- the scene can be regenerated from a seed.

With a real sequence, an error can come from bad calibration, bad matches,
wrong scale, or evaluation mistakes. With the synthetic scene those variables
are separated. The same code is then run on real frames through
`scripts/run_bookshelf_demo.py` and `scripts/run_sequence.py`.

---

## 2. Scene model

`vslam/simulation.py`

A scene is a list of **textured planar patches**. Each patch stores:

- a centre `C` in world coordinates,
- two half-axis vectors `u` and `v`,
- a texture image.

Its four corners are

```text
C - u - v,  C + u - v,  C + u + v,  C - u + v
```

which correspond to the texture corners `(0,0)`, `(w,0)`, `(w,h)`, `(0,h)`.
A patch can be a small floating panel, a wall, a floor, or a poster; the
renderer does not care.

The default `make_scene` creates 90 panels in a volume roughly 6 m wide,
4 m high and 4 m deep. Each panel has a random muted texture generated from:

- a low-frequency random grayscale field,
- random rectangles, circles and lines,
- a single random colour tint,
- a small amount of Gaussian noise.

The patterns are unique per panel, which gives SIFT distinctive descriptors
without making the images look like rainbow noise.

---

## 3. Camera path

The camera follows a smooth arc around the scene centre:

```text
angle(t)  = -26° … +26°
radius(t) = 5.4 + 0.35 sin(2πt)
eye(t)    = target + (r sin θ, 0.18 sin 2πt, -r cos θ)
```

At every frame the camera looks at the scene centre. `look_at` builds the
camera-to-world transform

```text
T_wc = [ right  down  forward  eye ]
       [   0      0      1      0  ]
```

with OpenCV's convention: `x` right, `y` down, `z` forward. The path has both
rotation and translation, so the epipolar geometry is well constrained and
the scene is not a single plane.

The default sequence has 24 frames. The accumulated optical-flow selector
picks 9 keyframes, which is enough to show drift and refinement without
making the regression test slow.

---

## 4. Rendering

For a camera pose `T_wc`, the renderer computes the world-to-camera transform

```text
T_cw = T_wc⁻¹
```

and projects each patch corner:

```text
X_c = R_cw X_w + t_cw
u   = fx X_c.x / X_c.z + cx
v   = fy X_c.y / X_c.z + cy
```

A patch is culled when any corner is behind the camera or when its projected
bounding box is outside the image. Otherwise, the four texture corners and
the four projected corners define a homography, and `cv2.warpPerspective`
renders the texture into the image.

Patches are drawn **far to near** (the painter's algorithm). This is enough
for the non-intersecting panels used here. The renderer is not a z-buffer
renderer, so intersecting geometry or large overlapping planes would need a
depth test. Gaussian noise is added at the end and the image is clipped to
`uint8`.

There is no lens distortion in the synthetic images; the camera model supports
distortion for real sequences. There is no rolling shutter, motion blur,
vignetting or exposure change.

---

## 5. Outputs

`generate_dataset` writes:

```text
dataset/
  images/frame_000.jpg …
  camera.json          # fx, fy, cx, cy, width, height, dist
  ground_truth.csv     # frame, x, y, z, r00 … r22
```

The ground-truth CSV stores the camera centre and the rotation matrix for each
frame. The synthetic runner converts it into the reconstruction world frame
(the first selected keyframe) before comparing:

```text
T_wc_relative(i) = T_wc(0)⁻¹ · T_wc(i)
```

This step is essential: the simulator's world frame and the SLAM world frame
are not the same. The first keyframe always has the identity pose in SLAM, so
comparing raw simulator poses to the estimate would report a large constant
offset.

---

## 6. Evaluation protocol

The estimated camera centres are aligned to ground truth with Umeyama's
closed-form Sim(3) solution. This removes the arbitrary monocular scale and
the global rotation/translation gauge. Then:

- **ATE** is the RMSE of the aligned camera centres,
- **rotation RPE** is the mean geodesic angle between consecutive relative
  rotations,
- **translation RPE** is the mean norm of the difference between consecutive
  relative translations after alignment,
- **RMS reprojection error** is computed over every map observation.

The reported numbers are therefore "shape and relative motion" errors, not
metric errors. A monocular system cannot recover metric scale without extra
information.

---

## 7. Reproducing the result

```bash
python scripts/run_synthetic_slam.py
```

Useful switches:

| Switch | Effect |
|---|---|
| `--seed 7` | change the scene texture and layout |
| `--frames 24` | number of rendered frames |
| `--patches 90` | scene density |
| `--noise 2.0` | image noise standard deviation |
| `--keyframe-threshold 8` | keyframe spacing |
| `--max-keyframes 10` | cap on keyframes |
| `--refine-iterations 3` | alternating BA sweeps |
| `--no-refine` | disable refinement |
| `--reuse-dataset` | reuse previously rendered images |

The output directory contains the images, the ground truth, the estimated
trajectory, the map plot, the match contact sheet and `metrics.json`. These
synthetic images are intentionally not used in the project showcase; the
README uses the real bookshelf reconstruction.

---

## 8. Limitations

The simulator is intentionally simple. It does not model:

- occlusion between intersecting geometry (painter's algorithm only),
- shadows, reflections, transparency or specularities,
- lens distortion, rolling shutter or motion blur,
- exposure changes between frames,
- dynamic objects.

Those effects are exactly the ones that make real SLAM difficult. The
synthetic harness is a correctness and regression test for the geometry, not
a benchmark of robustness.

The design is also deliberately simulator-agnostic: there is no robot model,
no scene file, no bridge process, and no dependency on an external robotics
package. The whole scene is ordinary NumPy and OpenCV code.
