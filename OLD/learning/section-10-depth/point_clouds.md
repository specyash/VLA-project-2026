# Point Clouds

**Goal:** Understand 2D depth → 3D points, deprojection, and how optional spatial reasoning uses them in this repo.

**Prerequisites:** [realsense_pipeline.md](realsense_pipeline.md), [camera_calibration.md](../section-09-camera-calibration/camera_calibration.md)

**Next:** [Section 11 — YOLO](../section-11-yolo/yolo_fundamentals.md)

**Official references:**
- [RealSense rs2_deproject_pixel_to_point](https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html#pyrealsense2.rs2_deproject_pixel_to_point)

---

## 1. Intuition: depth image → cloud of 3D dots

A depth map is still a **grid**. A **point cloud** is a list of points in 3D:

```
Pixel (u, v) + depth d  →  Point (X, Y, Z) in camera frame
```

Visually:

```
        Z (forward)
        │
        │   •  •
        │ •      •     each • = one surface point
        └────────── X
       /
      Y
```

Robots use clouds for collision avoidance, 3D segmentation, and spatial reasoning.

---

## 2. Deprojection (pinhole + depth)

Given camera **intrinsics** (fx, fy, cx, cy) and depth **d** at pixel (u, v):

Conceptually:

```
X = (u - cx) * d / fx
Y = (v - cy) * d / fy
Z = d
```

(Exact convention depends on camera frame definition; RealSense SDK handles this.)

**RealSense one-liner:**

```python
import pyrealsense2 as rs

intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
depth_m = depth_frame.get_distance(u, v)  # meters
pt = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth_m)
# pt = [X, Y, Z] in meters, camera frame
```

---

## 3. Building a downsampled cloud (repo pattern)

From `automated_pipeline_vlm.py`:

```python
stride = 20
for y in range(0, 480, stride):
    for x in range(0, 640, stride):
        dist = depth_frame.get_distance(x, y)
        if 0.1 < dist < 2.5:
            pt = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], dist)
            points.append(pt)
```

**Why stride?** Full 1280×720 = ~900k points — too slow to plot every frame. Every 20th pixel ≈ manageable scatter plot.

**Why distance filter?** Ignore invalid/near/far noise.

---

## 4. Colored point clouds

If you also have RGB:

```
color[u,v] + deproject(u,v) → colored point (X,Y,Z,R,G,B)
```

Useful for human inspection and learning-based 3D models.

---

## 5. ASCII: from detection to 3D

```
YOLO center (cx, cy)
        │
        ▼
depth_frame.get_distance(cx, cy)  or  depth_img[cy, cx]
        │
        ▼
deproject → (X, Y, Z) camera
        │
        ▼
T_cam_to_robot  (if calibrated)
        │
        ▼
robot frame target
```

**v5 today:** stops at depth readout in HUD (mm), does not deproject for arm commands.

---

## 6. Uses in robotics (general)

| Application | How cloud helps |
|-------------|-----------------|
| Obstacle avoidance | Points above table height |
| Bin filling level | Height statistics in bin ROI |
| Grasp verification | Cloud difference before/after pick |
| VLM spatial context | 3D plot alongside language |

---

## 7. How this appears in the robotics repo

| File | Point cloud use |
|------|-----------------|
| `automated_pipeline_vlm.py` | Matplotlib 3D scatter from deprojected samples |
| `realsense_view.py` | Depth filtering / visualization experiments |
| `automated_pipeline_v5.py` | Single-point depth at object center (2D index, not full cloud) |
| `Arm_to_object/object_pick.py` | 3D poses from markers, not dense cloud |

---

## 8. Common beginner confusion

1. **depth_img[cy,cx] vs get_distance** — units and invalid handling may differ; be consistent.
2. **Camera vs robot frame** — deproject gives camera; need extrinsics for arm.
3. **Sparse cloud** — stride misses small objects; dense is costly.
4. **Mirrors and glass** — false points in cloud.

---

## 9. Practice exercises

1. Deproject center pixel; move object closer — watch Z change.
2. Build a 1000-point cloud; visualize with `matplotlib` 3D scatter.
3. Compute median depth inside a YOLO bounding box (robust vs single pixel).

---

## 10. Summary

Point clouds are the **3D lift** of depth images. This repo demonstrates deprojection in the **VLM demo**; the main sorter uses **2D homography + fixed Z**. Understanding clouds is the bridge to depth-based grasping and verification.

**Next:** [yolo_fundamentals.md](../section-11-yolo/yolo_fundamentals.md)
