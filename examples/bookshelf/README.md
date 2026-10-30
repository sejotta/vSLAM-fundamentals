# Bookshelf example sequence

Five real keyframes from a handheld pan across a bookshelf. The images are
640×480 JPEG frames; the camera was calibrated with:

```text
fx = 641.84, fy = 641.17
cx = 311.14, cy = 244.66
dist = [-0.0233, 0.2523, 0, 0, -0.5219]
```

`camera.json` stores the same values in a form the demo can read. These are
the five keyframes used by the original reconstruction; consecutive views
share enough texture for SIFT matching while still providing useful parallax.
They contain no people or identifying information.

```bash
python scripts/run_bookshelf_demo.py
```

The script undistorts each image, treats all five frames as keyframes, runs
the incremental PnP + triangulation pipeline, and writes the trajectory,
the coloured point cloud and a match contact sheet to `runs/bookshelf/`.
