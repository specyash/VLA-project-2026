# Pose Estimation with Fiducial Markers

**Goal:** Understand 6-DOF pose (position + orientation) of markers in the camera frame, and how that differs from homography-only mapping.

**Prerequisites:** [aruco_markers.md](aruco_markers.md), [Section 9 — Camera calibration](../section-09-camera-calibration/camera_calibration.md)

**Official references:**
- [OpenCV `estimatePoseSingleMarkers`](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [OpenCV `solvePnP`](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

---

## 1. Intuition: from “where in the image” to “where in 3D”

**Homography (Section 7)** answers: *given a point on the table plane, where is it on the robot XY plane?*

**Pose estimation** answers: *where is this marker in 3D relative to the camera?*

```
Homography:     pixel (u,v)  →  table (X,Y)     [2D→2D on a plane]
Pose estimate:  marker       →  (X,Y,Z, roll,pitch,yaw) in camera frame [6-DOF]
```

You need pose when:
- The table is **not** the only plane of interest.
- You want **camera-to-robot** calibration in 3D (hand-eye).
- Object tags sit at **known height** above the table.

---

## 2. What is 6-DOF pose?

| Component | Meaning |
|-----------|---------|
| **Translation** (x, y, z) | Position of marker center in meters or mm |
| **Rotation** (roll, pitch, yaw) or rotation matrix | How the marker is tilted |

**rvec** (rotation vector) + **tvec** (translation vector) are OpenCV’s compact form from `solvePnP`.

**Analogy:** Homography tells you the address on a map; pose tells you which building floor and which way the building faces.

---

## 3. What you must know: marker physical size

Pose scale depends on **real marker width** in meters:

```python
marker_length = 0.05  # 5 cm — must match printed size
rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    corners, marker_length, camera_matrix, dist_coeffs
)
```

Wrong size → wrong depth (Z) by a scale factor.

---

## 4. Camera intrinsics required

Pose uses the pinhole camera model:

- **K** — 3×3 camera matrix (focal length, principal point)
- **dist** — distortion coefficients

Without calibration, pose is geometrically inconsistent. See [camera_calibration.md](../section-09-camera-calibration/camera_calibration.md).

`Camera_Caliberation/Camera_caliberation.py` writes `camera_calibration.yaml` for this project.

---

## 5. OpenCV API (minimal example)

```python
import cv2
import cv2.aruco as aruco
import numpy as np

# K, dist from calibration YAML
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
dist = np.zeros((5, 1))  # example

dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
corners, ids, _ = detector.detectMarkers(gray)

if ids is not None:
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        corners, 0.05, K, dist
    )
    for i in range(len(ids)):
        cv2.drawFrameAxes(frame, K, dist, rvecs[i], tvecs[i], 0.03)
```

**`drawFrameAxes`** visualizes X (red), Y (green), Z (blue) on the marker — essential for debugging.

---

## 6. From camera pose to robot pose

Pipeline used in `Arm_to_object/object_pick.py`:

```
1. Detect tag → pose in camera frame (tvec, rvec)
2. Apply hand-eye transform T_cam_to_robot (from calibration)
3. Get tag position in robot base frame (mm)
4. Move arm to hover above that XY, fixed Z offset
```

Calibration file: `Arm_to_object/tag_robot_mapping.yaml` stores:
- Correspondences: tag ID ↔ robot saved reading
- `transform_camera_to_robot` (rotation + translation)

**Note:** That YAML reports high mean error (~214 mm) in the checked-in file — recalibration may be needed before trusting 3D picks.

---

## 7. Homography path vs pose path (this repo)

| Approach | Used in | Needs intrinsics? | Best when |
|----------|---------|-------------------|-----------|
| **Homography + tag centers** | `automated_pipeline_v5.py` | No (planar fit) | Flat table, fixed camera |
| **Pose + 3D transform** | `Arm_to_object/object_pick.py` | Yes | General 3D, moving camera |

Main pipeline chose homography for **speed and simplicity** on a flat workspace.

---

## 8. ASCII: pose in full stack

```
Marker corners (2D pixels)
        │
        ▼
 estimatePoseSingleMarkers + K, dist
        │
        ▼
 tvec, rvec (camera frame, meters)
        │
        ▼
 T_cam_to_robot  (from calibration session)
        │
        ▼
 Robot (X, Y, Z) mm  →  arm.set_position(...)
```

---

## 9. How this appears in the robotics repo

- **v5:** Does **not** call `estimatePoseSingleMarkers`; uses 2D centers only.
- **`Pos_on_april_tags.py`:** Operator places tool on each corner tag; records robot pose → `ROBOT_READINGS`.
- **`object_pick.py`:** Full PnP + safety checks + robot motion.

---

## 10. Common beginner confusion

1. **Pose Z vs pick Z** — Marker pose Z is relative to camera; arm uses table `Z_LIMIT` separately in v5.
2. **rvec units** — Rotation vector; use OpenCV helpers to convert to matrix if needed.
3. **Single marker vs four** — Homography needs 4 coplanar points; pose works per marker if size and K are known.
4. **Glare / motion blur** — Pose jitters; temporal filtering helps.

---

## 11. Practice exercises

1. After calibrating camera, draw axis on one tag and rotate the tag — watch axes rotate.
2. Compare homography XY vs pose-projected XY for the same tag center (should be similar on a flat table).
3. Read `tag_robot_mapping.yaml` and identify which correspondence looks wrong.

---

## 12. Summary

- **Pose estimation** gives 6-DOF marker geometry in 3D using **known marker size** and **camera calibration**.
- **Main sorting pipeline** uses simpler 2D homography; **Arm_to_object** shows the full 3D path.
- For reliable 3D picks, invest in **good intrinsics** and **hand-eye calibration**, not only homography.

**Next:** [Section 9 — Camera calibration](../section-09-camera-calibration/camera_calibration.md)
