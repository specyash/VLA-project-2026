# Camera Basics

**Goal:** Understand how cameras form images and why calibration exists.

**Prerequisites:** [image_representation.md](image_representation.md)

**Next:** [Section 3 — OpenCV basics](../section-03-opencv-fundamentals/opencv_basics.md)

---

## 1. Intuition: pinhole camera

Light rays from a 3D point pass through a small **hole** (lens simplified) and hit the **image plane**.

```
        Object ●
              /|
             / |
            /  |
    Lens ---   ●  image on sensor (2D)
```

**Perspective:** farther objects project smaller — same as how a road narrows in photos.

---

## 2. 3D world → 2D image

Each 3D point `(X, Y, Z)` in the **camera frame** maps to pixel `(u, v)`:

- **Intrinsic** parameters: focal length, principal point, distortion (lens bend)
- **Extrinsic** pose: where the camera sits relative to the robot/table

Full formula (intuition only for now):

```
pixel = project( camera_matrix × [R|t] × world_point )
```

**Why robotics cares:** The arm lives in **robot mm**; the camera sees **pixels**. You must bridge the two (homography in this repo for a flat table, or full calibration in `Camera_Caliberation/`).

---

## 3. Camera frame

Axes (typical computer vision):

- **Z** forward (into the scene)
- **X** right in the image
- **Y** down in the image

The **image plane** is where pixels live; it is not the same as the robot’s base frame.

---

## 4. Why depth is lost

One pixel `(u, v)` could be:

- a near small object, or
- a far large object

Same ray direction — different distances. RGB alone cannot disambiguate without extra geometry (depth sensor, stereo, or known object size).

---

## 5. RealSense in this project

Intel RealSense gives:

- **Color** stream (BGR)
- **Depth** stream (distance per pixel, mm)

Aligned so color pixel `(cx, cy)` matches depth at the same index:

```python
obj_depth_mm = depth_img[obj["cy"], obj["cx"]]
```

Used in the **topographic HUD**, not for pick height (picks use fixed `Z_LIMIT` + offsets).

---

## 6. Webcam fallback

If `pyrealsense2` is missing, `get_video_stream()` uses `cv2.VideoCapture(0)` — RGB only, no depth HUD data.

---

## How this appears in the robotics repo

- `get_video_stream()` — RealSense 1280×720 @ 30 FPS or webcam
- `rs.align(rs.stream.color)` — depth registered to color
- `ROBOT_READINGS` + homography — bypass full 3D projection for flat tabletop

---

## Common beginner confusion

- **Megapixels ≠ robot accuracy** — lens quality, calibration, and mounting matter more.
- **Camera tilt** — changes homography; tags help re-estimate the plane live.
- **Depth ≠ always correct** — shiny/black surfaces return invalid zero depth.

---

## Practice

1. Draw side-view sketch: camera above table, one cube close and one far — which looks bigger?
2. Open `automated_pipeline_v5.py` and find `get_video_stream()` — list what streams start.

---

## Summary

Cameras **project 3D → 2D**. Robotics needs **inverse** mapping (2D → 3D or 2D table plane). This repo uses **planar homography** + **AprilTags** for that on a flat workspace.
