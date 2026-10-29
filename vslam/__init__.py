"""A compact visual SLAM toolkit built around OpenCV and NumPy.

The package is intentionally small and readable.  It implements the
fundamentals that a visual odometry / SLAM front-end needs:

* feature detection and matching,
* epipolar geometry and relative pose,
* triangulation and PnP,
* keyframe selection,
* incremental mapping, and
* a lightweight bundle-adjustment refinement.

The public entry points live in :mod:`vslam.pipeline` and
:mod:`vslam.simulation`.
"""

from .camera import Camera
from .pipeline import IncrementalSlam, SlamConfig, SlamResult

__all__ = ["Camera", "IncrementalSlam", "SlamConfig", "SlamResult"]
__version__ = "0.1.0"
