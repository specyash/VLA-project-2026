# Depth Cameras

**Goal:** Understand how robots recover 3D information, what a depth map is, and how this project uses (and does not use) depth for control.

**Prerequisites:** [camera_basics.md](../section-02-images-and-cv-foundations/camera_basics.md), [camera_calibration.md](../section-09-camera-calibration/camera_calibration.md)

**Next:** [realsense_pipeline.md](realsense_pipeline.md)

**Official references:**
- [Intel RealSense depth technology overview](https://www.intelrealsense.com/depth-camera-d435/)
- [librealsense docs](https://dev.intelrealsense.com/docs)

---

## 1. The core problem: RGB is flat

A normal color image records **color per pixel**, not distance.

Two objects can share the same pixel color but sit at very different depths — the robot cannot know which is closer without extra sensing.

**Depth sensing** adds a **distance channel**.

---

## 2. Intuition: measuring distance per pixel

Imagine shining a pattern or measuring time-of-flight for each pixel:

```
RGB image:     Depth image:
[red green]    [ 450mm  1200mm ]
[blue yellow]  [ 448mm  1195mm ]
```

Each pixel in the depth map stores distance to the surface (often in **millimeters** for RealSense Z16).

---

## 3. Common depth technologies

| Technology | Idea | Examples |
|------------|------|----------|
| **Stereo** | Two cameras + triangulation | RealSense D435 |
| **Structured light** | Projected pattern + IR camera | RealSense D415 |
| **Time-of-flight (ToF)** | Light round-trip time | Some industrial sensors |
| **LIDAR** | Scanning laser | Not in this repo |

This lab typically uses **Intel RealSense** (stereo or structured light depending on model).

---

## 4. Depth map properties

- **Same resolution** as color (after alignment) or different before align.
- **Invalid pixels** often stored as **0** — no return (too far, absorbent surface, glare).
- **Noise** at object edges and on dark/shiny materials.
- **Units:** RealSense Z16 in mm at sensor; can convert to meters for 3D math.

---

## 5. RGB vs depth vs “3D understanding”

| Modality | Gives you |
|----------|-----------|
| RGB | Color, texture, good for YOLO / markers |
| Depth | Metric distance per pixel |
| RGB-D together | Colored point cloud, obstacle maps |

**This repo (v5):**
- RGB → YOLO, ArUco, HSV marker check.
- Depth → **HUD visualization** and per-object depth readout text.
- **Pick height** still uses fixed `Z_LIMIT` + `bin_offsets`, not live depth.

That is a deliberate simplification — but an improvement opportunity.

---

## 6. Why depth still matters in your project

Even without closed-loop grasp Z:

1. **Operator trust** — HUD shows “this cube is ~430 mm away.”
2. **Debugging** — wrong table height or camera tilt shows in depth layers.
3. **Future features** — verify grasp (depth before/after), collision check, VLM spatial plots (`automated_pipeline_vlm.py`).

---

## 7. ASCII: depth in perception stack

```
RealSense
   ├── Color BGR  ──► YOLO, ArUco, UI
   └── Depth Z16  ──► align to color
                         │
                         ▼
                   depth[cy, cx]  ──► HUD label (mm)
                         │
            (not used in v5 for arm Z command)
```

---

## 8. How this appears in the robotics repo

```python
# automated_pipeline_v5.py — HUD only
obj_depth_mm = depth_img[obj["cy"], obj["cx"]]
```

```python
# create_techy_hud — clip, normalize, colormap, Canny edges
depth_clipped = np.clip(depth_small, 100, 2500)
```

If `pyrealsense2` missing → `depth_frame is None`; pipeline still runs on webcam RGB.

---

## 9. Common beginner confusion

1. **Depth ≠ height above table** — depth is along the **camera ray**, not vertical unless camera looks straight down.
2. **Zero depth** — invalid, not “zero distance.”
3. **Alignment required** — raw depth index ≠ color index without `rs.align`.
4. **Glass / black blocks** — often missing depth values.

---

## 10. Practice exercises

1. Point camera at your hand; print depth at center pixel while moving closer/farther.
2. Compare depth at two cubes at different distances — confirm ordering.
3. Note which surfaces return zero depth in your lab lighting.

---

## 11. Summary

Depth cameras recover **metric 3D structure**. This project uses depth primarily for **visualization and debugging**; manipulation Z is still **rule-based**. Understanding depth prepares you for grasp verification and true RGB-D picking.

**Next:** [realsense_pipeline.md](realsense_pipeline.md)
