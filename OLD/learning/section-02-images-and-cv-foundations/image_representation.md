# Image Representation (Matrices & Channels)

**Goal:** Represent images as NumPy arrays the way OpenCV does.

**Prerequisites:** [how_images_work.md](how_images_work.md)

**Next:** [camera_basics.md](camera_basics.md)

---

## 1. Intuition: spreadsheet of colors

A color image is like **three stacked spreadsheets** (B, G, R). Each cell holds 0–255.

```
Channel B    Channel G    Channel R
[  120 ]     [  80  ]     [  200 ]  → looks reddish in BGR
```

**Shape:** `(height, width, 3)` — in code: `frame.shape` → `(720, 1280, 3)` for 720p.

---

## 2. NumPy representation

```python
import numpy as np
import cv2

# Synthetic 100×200 blue-ish patch (BGR)
img = np.zeros((100, 200, 3), dtype=np.uint8)
img[:, :] = (255, 0, 0)  # B=255, G=0, R=0

print(img.shape)   # (100, 200, 3)
print(img.dtype)   # uint8 → integers 0-255
```

**Why `uint8`?** Standard for camera frames; saves memory vs floats.

---

## 3. Indexing and ROIs

```python
# One pixel (y, x)
b, g, r = img[50, 100]

# Region of interest (ROI) — crop around detection
y1, y2, x1, x2 = 100, 200, 50, 150
roi = frame[y1:y2, x1:x2]
```

**Repo:** `is_marked_with_black(frame[y1:y2, x1:x2])` crops each YOLO box to test for black marker.

---

## 4. Grayscale as 2D

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# shape: (height, width) — no channel axis
```

---

## 5. Binary masks

A **mask** is an image where each pixel is 0 or 255 (off/on).

```python
mask = np.zeros((h, w), dtype=np.uint8)
mask[100:150, 200:250] = 255
# Only white region is "selected"
```

Used in color segmentation (Section 4) and `cv2.inRange`.

---

## 6. Memory layout (intuition)

Large frames are contiguous blocks in RAM. Copying matters:

```python
frame_copy = frame.copy()  # safe for another thread
frame_ref = frame          # same data if modified elsewhere
```

`YoloWorker.push()` uses `frame.copy()` so the worker does not race with the main loop.

---

## How this appears in the robotics repo

| Operation | Location |
|-----------|----------|
| `frame.shape` | Main loop after `read_frame()` |
| ROI crop | `is_marked_with_black`, YOLO boxes |
| `np.array(..., dtype=np.float32)` | Workspace polygon, homography points |
| `np.clip`, `np.normalize` | Depth HUD (`create_techy_hud`) |

---

## Common beginner confusion

- **`img[x,y]` vs `img[y,x]`** — always row (y) first in NumPy.
- **Modifying ROI modifies original** — use `.copy()` when sending to threads.
- **Float vs uint8** — homography uses `float32`; display uses `uint8`.

---

## Practice

1. Create a 50×50 red square in the center of a 200×200 BGR image.
2. Given YOLO box `(x1,y1,x2,y2)`, write code to compute `cx, cy`.

---

## Summary

Images are **NumPy arrays**: 2D for gray/masks, 3D for BGR. Robotics crops **ROIs** around detections and passes **float** point arrays to geometry functions.
