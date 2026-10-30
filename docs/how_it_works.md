# How it works

This document explains the geometry and the design decisions behind the
pipeline. It is written to be read alongside the code: every section points to
the module that implements it.

The core of the reconstruction is the loop described in Section 6: **PnP
localises each new keyframe against the 3D points already reconstructed, and
triangulation adds the new structure observed in that keyframe**. The
synthetic scene mentioned in Section 10 is a deterministic regression harness,
not the project's input data.

---

## 1. Frames and notation

A 3D point in world coordinates is `X_w`. A camera pose is a rigid transform

```text
T_wc = [ R_wc   t_wc ]      X_w = R_wc X_c + t_wc
       [  0      1  ]
```

which maps **camera coordinates to world coordinates**. The inverse

```text
T_cw = T_wc⁻¹ = [ R_wcᵀ   -R_wcᵀ t_wc ]
                [   0           1     ]
```

maps **world coordinates to camera coordinates** and is the pose returned by
OpenCV's PnP functions. `recoverPose` returns a relative transform `T_21`
between two cameras instead; the two conventions are kept separate throughout
the code.

The first keyframe is the world origin, so `T_wc(0) = I`. Monocular
reconstruction has no metric scale: multiplying every translation by a
constant produces exactly the same images. When ground truth is available,
accuracy comparisons therefore use a Sim(3) alignment (Section 10). The
bookshelf example has no ground-truth trajectory, so only its reprojection
error is reported.

The camera is a pinhole model with intrinsics

```text
K = [ fx  0  cx ]
    [  0 fy  cy ]
    [  0  0   1 ]
```

and OpenCV distortion coefficients `(k1, k2, p1, p2, k3)`.

---

## 2. Feature detection and matching

`vslam/features.py`

The default back-end is SIFT:

1. convert the RGB image to grayscale,
2. detect up to `max_features` keypoints and compute 128-D descriptors,
3. match descriptors with a brute-force L2 matcher,
4. keep a pair only if

```text
d(first neighbour) < ratio · d(second neighbour)
```

with `ratio = 0.75`. This is Lowe's ratio test: ambiguous descriptors are
discarded instead of being forced into a match.

The optional `SuperPointSuperGlueMatcher` replaces both steps with the public
SuperPoint detector and SuperGlue graph neural network. It exposes exactly the
same interface, so the geometry code does not change.

A match is stored as an integer pair `(index_a, index_b)`. All downstream
stages work with these indices; pixel coordinates are looked up only when
needed.

---

## 3. Keyframe selection

`vslam/keyframes.py`

Using every frame is wasteful: consecutive frames have a tiny baseline and
triangulation from near-parallel rays is ill-conditioned. The selector tracks
corners between consecutive frames:

1. `cv2.goodFeaturesToTrack` finds up to 300 corners in the previous frame.
2. `cv2.calcOpticalFlowPyrLK` tracks them into the current frame.
3. The mean displacement of successfully tracked corners is accumulated.
4. When the accumulator reaches `translation_threshold`, the current frame
   becomes a keyframe and the accumulator resets.
5. If there are more candidates than `max_keyframes`, they are uniformly
   subsampled.

The threshold is a trade-off:

- too small → many near-duplicate keyframes, little parallax, noisy
  triangulation;
- too large → few keyframes, less overlap, more matching failures.

The generic sequence runner uses `25 px`; the bookshelf demo supplies five
frames that are already treated as keyframes, so the selector is bypassed.

---

## 4. Two-view geometry

`vslam/geometry.py`

### 4.1 Fundamental matrix

For corresponding image points `x1` and `x2` (homogeneous),

```text
x2ᵀ F x1 = 0
```

`F` is 3×3, rank 2, and has 7 degrees of freedom. The implementation uses
OpenCV's 8-point algorithm inside RANSAC:

```python
F, mask = cv2.findFundamentalMat(
    points1, points2, cv2.FM_RANSAC, ransacReprojThreshold=0.8
)
```

The threshold is the Sampson/epipolar distance in pixels. The mask marks the
correspondences that agree with the dominant epipolar geometry.

### 4.2 Essential matrix

With a calibrated camera,

```text
E = Kᵀ F K
```

`E` encodes the same geometry in normalised camera coordinates. It has two
equal non-zero singular values and one zero singular value. The code
normalises `E` by its Frobenius norm for numerical stability; the scale is
irrelevant.

### 4.3 Relative pose

`cv2.recoverPose` decomposes `E` into four candidate solutions
`(R, ±t)` and keeps the one for which the triangulated points are in front of
both cameras (the **cheirality check**). The result is

```text
X_2 = R X_1 + t        i.e.  T_21 = [R | t]
```

where `T_21` maps camera-1 coordinates to camera-2 coordinates.

### 4.4 Triangulation

With camera 1 at the origin,

```text
P1 = K [I | 0]
P2 = K [R | t]
```

Each correspondence contributes two linear equations. Stacking all equations
and taking the last singular vector of the SVD gives the homogeneous 3D point;
dividing by its fourth coordinate gives `X = (X, Y, Z)`.

The implementation calls `cv2.triangulatePoints` and normalises the result.
Triangulation is only accepted when:

- the point is in front of both cameras (`z > min_depth`),
- it is not beyond `max_depth` (near-parallel rays produce points at
  infinity),
- its reprojection error in both views is below `max_reproj_error`,
- the angle between the two viewing rays is above
  `min_triangulation_angle` (a small angle means a poorly conditioned
  baseline).

---

## 5. Initialisation

`IncrementalSlam._initialize` in `vslam/pipeline.py`

The first two keyframes bootstrap the map:

1. detect features in both frames and match them,
2. estimate `F` with RANSAC,
3. compute `E = Kᵀ F K` and recover `T_21`,
4. triangulate the inlier matches in camera-1 coordinates,
5. apply the depth / reprojection / angle filters,
6. set `T_wc(0) = I` and `T_wc(1) = T_21⁻¹`,
7. store each 3D point with its colour sampled from the first image.

If any stage produces too few matches, inliers or valid points, the run stops
with a descriptive error. Initialisation is the most fragile part of a
monocular pipeline; a sequence that starts with a pure rotation or a planar
scene may not provide enough parallax.

---

## 6. Tracking a new keyframe with PnP

`IncrementalSlam._add_keyframe`

This is the incremental reconstruction step: the existing map is used to
localise the new camera, and the new camera is used to extend the map.

For each new keyframe `k`:

1. **Match** the previous keyframe `k-1` to `k`.
2. **RANSAC** estimates `F` and keeps the inlier matches.
3. **Separate** the inliers into two groups:
   - matches whose previous keypoint already has a 3D map point,
   - matches that have no 3D point yet.
4. **PnP** on the first group:

   ```python
   ok, rvec, tvec, inliers = cv2.solvePnPRansac(
       object_points_3d, image_points_2d, K, dist,
       flags=cv2.SOLVEPNP_ITERATIVE,
   )
   ```

   `solvePnPRansac` returns `T_cw`, the world-to-camera pose. The camera pose
   in the world is `T_wc = T_cw⁻¹`. Inlier matches propagate their map-point
   index to the current keyframe, which is what makes later PnP and
   triangulation possible.
5. **Fallback**: if too few 3D-2D correspondences are available, recover the
   relative pose from `E` and compose it:

   ```text
   T_wc(k) = T_wc(k-1) · T_21⁻¹
   ```

6. **Triangulate** the second group using the two camera poses. The same
   depth / reprojection / angle filters are applied. Accepted points are
   transformed to world coordinates, coloured from the previous image, and
   added to the map.
7. **Associate** every accepted point with both keypoint indices and append
   two observations for bundle adjustment.

The map is therefore a set of 3D points plus a many-to-many association table
between map points and keypoint detections.

---

## 7. Fallback pose composition and drift

The primary localisation path is PnP. When too few 3D-2D correspondences are
available, the pipeline falls back to composing the relative motion from `E`:

```text
T_wc(k) = T_wc(k-1) · T_21(k)⁻¹
```

Even with PnP, small rotation and translation errors accumulate over the
sequence. This is the classical **drift** of visual odometry. The pipeline
reduces it with local refinement, but it does not perform loop closure or
global pose-graph optimisation, so long sequences will bend.

---

## 8. Alternating bundle adjustment

`IncrementalSlam._refine`

A full bundle adjustment would optimise all camera poses and all 3D points
jointly with a sparse Levenberg–Marquardt solver. To keep the project
dependency-light, the code uses a block-coordinate approximation:

1. **Fix the points, refine the cameras.** For every camera except camera 0,
   collect its 2D observations, solve PnP with the current pose as the initial
   guess, and replace the pose if PnP succeeds.
2. **Fix the cameras, refine the points.** For every map point observed by at
   least two cameras, re-triangulate it with the DLT using all of its
   observations. Keep the new point only if it is in front of at least two
   cameras.
3. Repeat for a few sweeps.

Camera 0 stays fixed to remove the gauge freedom (the world frame is defined
by it). This is not a substitute for a real BA solver, but it is simple,
readable, and measurably reduces reprojection error and drift on the demo.

---

## 9. Planar homography and mosaic

`vslam/homography.py`

When the scene is planar or the camera only rotates, the relation between two
views is a homography:

```text
x2 ~ H x1
```

`H` is 3×3, defined up to scale, and needs at least four correspondences.
`estimate_homography` uses `cv2.findHomography` with RANSAC (5 px threshold).

`warp_pair` projects the four corners of image 1 through `H`, computes a
canvas that contains both images, and warps image 1 into the plane of image 2.
The new image is copied on top with a clean seam (no blending). Empty regions
stay black; `crop_black_borders` removes the outer margins.

The self-contained homography test uses a single textured wall, so the planar
model is exact. The bookshelf mosaic shown in the README was produced from the
real sequence by the original prototype. A real scene with depth violates the
planar model: parallax makes the frames disagree and the mosaic drifts.

---

## 10. Evaluation

`vslam/evaluate.py`

Monocular estimates are only defined up to a similarity transform. Before
computing errors, the estimated camera centres are aligned to ground truth
with Umeyama's closed-form Sim(3) solution:

```text
aligned = s · R · estimate + t
```

The scale `s`, rotation `R` and translation `t` are obtained from the SVD of
the cross-covariance between the two point sets. A reflection correction
ensures `det(R) = +1`.

Two metrics are reported:

- **ATE** — absolute trajectory error, the RMSE of the aligned camera centres.
- **RPE** — relative pose error between consecutive keyframes. Rotation error
  is the geodesic angle between the relative rotation matrices; translation
  error is the norm of the difference between the aligned relative
  translations.

The bookshelf sequence has no ground-truth trajectory, so only the RMS
reprojection error is reported for it. The synthetic regression scene has
exact poses and is the only place where ATE and RPE are computed.

---

## 11. Parameters

| Parameter | Bookshelf demo | Meaning |
|---|---:|---|
| `f_ransac_threshold` | 0.8 px | epipolar inlier threshold |
| `max_reproj_error` | 2.5 px | triangulation reprojection gate |
| `max_depth` | 50.0 | reject points beyond this depth |
| `min_depth` | 0.05 | reject points behind/too close |
| `min_triangulation_angle` | 0.5° | reject poorly conditioned rays |
| `pnp_reprojection_threshold` | 4.0 px | PnP RANSAC threshold |
| `min_pnp_points` | 6 | minimum 3D-2D correspondences |
| `refine_iterations` | 3 | alternating BA sweeps |

The bookshelf demo uses its five supplied frames directly. The generic
`run_sequence.py` path uses `keyframe_threshold=25 px` and
`max_keyframes=10` by default; those values are a better starting point for a
long video.

---

## 12. When it fails

- **Pure rotation / planar scene.** The fundamental matrix is degenerate or
  poorly constrained; use the homography path instead.
- **Too little texture.** SIFT finds few stable keypoints; a learned detector
  or a different sensor may help.
- **Large exposure or viewpoint changes.** Matching fails; SuperPoint +
  SuperGlue is more robust than SIFT here.
- **Fast motion / motion blur.** Optical flow and matching both degrade.
- **Long sequences.** Drift accumulates without loop closure.
- **Wrong intrinsics.** The essential matrix and PnP become inconsistent;
  calibration matters.

---

## 13. References

1. R. Hartley and A. Zisserman, *Multiple View Geometry in Computer Vision*,
   Cambridge University Press, 2003.
2. D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints",
   *International Journal of Computer Vision*, 2004.
3. D. DeTone, T. Malisiewicz and A. Rabinovich, "SuperPoint: Self-Supervised
   Interest Point Detection and Description", *CVPR Workshops*, 2018.
4. P.-E. Sarlin, D. DeTone, T. Malisiewicz and A. Rabinovich, "SuperGlue:
   Learning Feature Matching with Graph Neural Networks", *CVPR*, 2020.
5. S. Umeyama, "Least-squares estimation of transformation parameters between
   two point patterns", *IEEE TPAMI*, 1991.
