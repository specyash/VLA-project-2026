# How Images Work

**Goal:** Understand what a digital image *is*, before using OpenCV.

**Prerequisites:** [Section 1 — Python basics](../section-01-python-foundations/python_basics_for_cv.md)

**Next:** [image_representation.md](image_representation.md)

---

## 1. Intuition: a image is a grid of numbers

Imagine standing above a chessboard. Each square has a color. A digital image is the same idea at **millions** of tiny squares called **pixels** (picture elements).

```
┌──┬──┬──┬──┐
│██│░░│██│░░│   Each cell = one pixel
├──┼──┼──┼──┤   "██" might mean "dark"
│░░│██│░░│██│   "░░" might mean "bright"
└──┴──┴──┴──┘
```

A computer does not “see” objects — it stores **numbers**. Vision algorithms interpret those numbers as edges, colors, and objects.

**Why this matters for robotics:** The robot arm needs **positions** derived from those numbers (pixel → millimeters on the table).

---

## 2. From light to file

```
Real world → Lens → Sensor → Array of numbers → .jpg / live frame
```

1. Light hits the sensor.
2. Each sensor cell records brightness (and color filters give R, G, B).
3. Software packs values into a **frame** (one snapshot in time).

Video is a **sequence of frames** (e.g. 30 per second).

---

## 3. What you lose immediately

A photo is a **2D flattening** of 3D space:

- Two objects at different depths can overlap in the image.
- Size in the image depends on **distance** (far objects look smaller).

Normal RGB cameras **do not** store per-pixel distance. That is why this repo also uses a **depth camera** (RealSense) for the HUD — see Section 10.

---

## 4. Resolution

**Resolution** = width × height in pixels.

| Name | Typical size | Use |
|------|--------------|-----|
| VGA | 640×480 | Fast tests |
| 720p | 1280×720 | This repo’s RealSense config |
| 1080p | 1920×1080 | Higher detail, slower |

More pixels → more detail and more computation for YOLO.

---

## 5. Coordinate system in images

```
(0,0) ──────────────> x (columns, width)
  │
  │
  ▼
  y (rows, height)
```

- **Origin** is top-left (OpenCV convention).
- **x** increases to the right.
- **y** increases **downward** (opposite of math class graphs).

Object center in the repo: `cx = (x1 + x2) // 2`, `cy = (y1 + y2) // 2`.

---

## 6. Color vs grayscale

- **Grayscale:** one number per pixel (0 = black, 255 = white).
- **Color (BGR in OpenCV):** three numbers per pixel — Blue, Green, Red channels.

ArUco detection in the repo converts to gray:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

Tags are high-contrast black/white patterns — color is unnecessary for detection.

---

## 7. ASCII: frame in the pipeline

```
Camera frame (BGR matrix)
        │
        ├─► grayscale ──► ArUco tags
        │
        ├─► BGR ──► YOLO ──► bounding boxes
        │
        └─► depth (if RealSense) ──► HUD
```

---

## How this appears in the robotics repo

- `read_frame()` returns `frame` as a NumPy array (`automated_pipeline_v5.py`).
- Object centers `(cx, cy)` are in **pixel coordinates**.
- `inside_workspace(px, py)` tests if those pixels lie inside the workspace polygon.

---

## Common beginner confusion

- **“The image looks fine on screen”** — display may auto-scale; processing uses raw array indices.
- **Confusing x/y with row/col** — OpenCV `img[y, x]` uses row first.
- **Expecting depth from a webcam** — standard webcams are RGB-only; depth needs stereo/ToF (RealSense).

---

## Practice

1. Sketch a 4×4 grid and label pixel `(2, 3)` in (x,y) and in (row, col).
2. Why does the repo run ArUco on grayscale but YOLO on color?

---

## Summary

An image is a **2D array of pixel values** over time. Robotics uses pixel locations as clues; later sections convert them to **robot coordinates**.
