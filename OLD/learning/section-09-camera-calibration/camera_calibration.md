# Camera Calibration

**Goal:** Understand intrinsics, distortion, extrinsics, and the calibration workflow used (and not yet wired) in this project.

**Prerequisites:** [Section 2 — Camera basics](../section-02-images-and-cv-foundations/camera_basics.md), [pose_estimation.md](../section-08-aruco/pose_estimation.md)

**Next:** [Section 10 — Depth cameras](../section-10-depth/depth_cameras.md)

**Official references:**
- [OpenCV camera calibration tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV `calibrateCamera`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga32076a1a41034518d9a0c6b270ac7346)

---

## 1. Why calibration exists

Real lenses **do not** match the ideal pinhole model:

- Lines that should be straight may curve (**distortion**).
- Focal length and sensor center are unknown without measurement.
- Pixel size alone does not give metric depth.

**Without calibration:**
- Pose from markers is scaled wrong.
- 3D deprojection from depth is wrong.
- “5 cm” in software ≠ 5 cm in reality.

**Course requirement:** intrinsic calibration, extrinsic (camera-to-robot), demonstrated image→robot mapping.

---

## 2. Intuition: teaching the camera its ruler

Calibration is like telling the camera:

> “When you see this pattern at these pixel locations, the real-world geometry looked **exactly like this**.”

You show a **chessboard** with known square size many times; OpenCV solves for internal parameters.

---

## 3. Intrinsic parameters (K matrix)

The **camera matrix** K (3×3):

```
K = | fx   0   cx |
    |  0  fy   cy |
    |  0   0    1 |
```

| Symbol | Meaning |
|--------|---------|
| **fx, fy** | Focal length in pixel units (horizontal / vertical) |
| **cx, cy** | Principal point — where optical axis hits the sensor (often near image center) |

**Effect:** Converts between **camera coordinates** (rays) and **pixels**.

---

## 4. Distortion coefficients

Typical model: radial (k1, k2, k3) + tangential (p1, p2).

**Symptoms if ignored:**
- Straight table edges look curved near image borders.
- ArUco corners shift slightly → homography and pose error at edges.

**Fix:** `cv2.undistort()` or pass `dist` into `solvePnP` / `estimatePoseSingleMarkers`.

---

## 5. Extrinsics (camera pose in world / robot frame)

**Extrinsic** = rotation **R** + translation **t** from world (or robot base) to camera.

Used for **hand-eye calibration**: camera mounted on arm or fixed beside robot — “where is the camera relative to the robot base?”

This repo has two extrinsic stories:

1. **Planar homography** in v5 (implicit extrinsics for the table plane only).
2. **`tag_robot_mapping.yaml`** in `Arm_to_object/` (explicit 3D transform).

---

## 6. Calibration workflow (chessboard)

From `Camera_Caliberation/Camera_caliberation.py`:

### Step A — Capture images

- Print chessboard (inner corners e.g. 7×5).
- Move board to many poses: center, corners, tilted, near/far.
- Save 15–30+ sharp images (`calib_000.png`, …).

### Step B — Detect corners

```python
found, corners = cv2.findChessboardCorners(gray, board_size)
```

Subpixel refine with `cornerSubPix` for accuracy.

### Step C — Solve

```python
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)
```

**objpoints:** 3D points on the board (Z=0 plane, spacing = square size in meters).

**imgpoints:** matching 2D corners per image.

### Step D — Evaluate

**Reprojection error:** project 3D points back to image; average pixel error should be **< 1 pixel** for good setups.

Script writes `camera_calibration.yaml` with K, dist, image size, RMS error.

---

## 7. Minimal code example

```python
import cv2
import numpy as np

# After calibrateCamera:
img = cv2.imread("test.jpg")
h, w = img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, K, dist, None, new_K)
```

Compare corners on raw vs undistorted — edges should align with straight lines.

---

## 8. Hand-eye calibration (concept)

When camera is **fixed** watching the table:

1. Touch each workspace AprilTag with the tool center (or pointer).
2. Record **robot (x,y,z)** and **tag pose in camera** (or pixel + depth).
3. Solve for transform **camera → robot**.

`Lite6_arm_pos/Pos_on_april_tags.py` records robot poses at tags.

`Arm_to_object/tag_robot_mapping.yaml` stores the solved transform (verify errors before use).

---

## 9. How this appears in the robotics repo

| Component | Calibration type | Used in v5? |
|-----------|------------------|-------------|
| `Camera_caliberation/` | Intrinsics + distortion | No (available for pose/depth) |
| `ROBOT_READINGS` + homography | Planar 4-point fit | **Yes** |
| `workspace_calibration.yaml` | Pixel fallback for tags | **Yes** |
| `tag_robot_mapping.yaml` | 3D camera→robot | `object_pick.py` only |

**Gap to be aware of:** Main demo does not load `camera_calibration.yaml`; accuracy relies on homography + manual corner teaching.

---

## 10. ASCII: calibration in the full system

```
Chessboard images ──► calibrateCamera ──► K, dist
                                              │
AprilTag poses + robot touches ──► hand-eye ──┼──► 3D picks (object_pick.py)
                                              │
4 tag pixels + 4 robot XY ──► homography ─────┘──► 2D picks (v5 main)
```

---

## 11. Common beginner confusion

1. **Intrinsics vs homography** — Intrinsics model the lens; homography models one plane. Different tools.
2. **Too few chessboard images** — High reprojection error; grasps miss.
3. **Wrong square size** — All depths scaled wrong.
4. **Calibration once forever** — Bump the camera → recalibrate or re-save tag homography.

---

## 12. Practice exercises

1. Run `Camera_caliberation.py` capture mode; save 20 images; note RMS error in output YAML.
2. Undistort one frame and overlay ArUco detection — compare corner stability.
3. List what breaks in v5 if you rotate the camera 10° without recalibrating.

---

## 13. Summary

- **Intrinsics** (K, dist) describe how the camera forms images.
- **Extrinsics** describe where the camera sits relative to the robot/world.
- This repo’s **main pipeline** uses homography for speed; full calibration supports **3D pose** and **depth deprojection** paths.

**Next:** [depth_cameras.md](../section-10-depth/depth_cameras.md)
