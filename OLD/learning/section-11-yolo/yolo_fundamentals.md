# YOLO Fundamentals

**Goal:** Understand object detection, YOLO’s “one-shot” idea, and Ultralytics YOLOv8 outputs used in this project.

**Prerequisites:** [Section 3 — OpenCV](../section-03-opencv-fundamentals/opencv_basics.md), [Section 4 — HSV](../section-04-color-spaces/hsv_and_color_detection.md)

**Next:** [object_detection_pipeline.md](object_detection_pipeline.md)

**Official references:**
- [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/)
- [YOLO object detection explained (Ultralytics)](https://docs.ultralytics.com/tasks/detect/)

---

## 1. What is object detection?

**Classification:** “What is in the image?” (one label per image)

**Detection:** “What objects are where?” → **boxes + class labels** per object

For sorting you need:
- **Where** is each block? → bounding box / center
- **What color class** is it? → YOLO class name (`red`, `green`, …)

---

## 2. Intuition: YOLO = one forward pass

Older pipelines: propose regions → classify each (slow).

**YOLO (You Only Look Once):** neural network divides image into a grid; each cell predicts boxes and classes in **one inference**.

```
Input image  →  CNN backbone  →  head predicts boxes + scores
```

Good for **real-time robotics** when GPU or strong CPU is available.

---

## 3. Key outputs (YOLOv8 / Ultralytics)

For each detection:

| Field | Meaning |
|-------|---------|
| **xyxy** | Box corners x1, y1, x2, y2 (pixels) |
| **cls** | Class index |
| **conf** | Confidence ∈ [0, 1] |
| **names** | Map index → string (`model.names`) |

Repo usage:

```python
res = model(frame, conf=YOLO_CONF, iou=0.45, verbose=False)[0]
for box in res.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    class_id = int(box.cls[0])
    base_color = model.names[class_id]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
```

**Model file:** `bestv8.pt` — custom-trained weights for colored blocks.

---

## 4. Confidence threshold (`YOLO_CONF = 0.40`)

Only detections with conf ≥ 0.40 are kept.

**Too high** → miss real objects.  
**Too low** → false boxes, wrong picks.

Tune on your table, lighting, and camera.

---

## 5. IoU and NMS (`iou=0.45`)

When two boxes overlap heavily, **Non-Maximum Suppression** keeps the stronger one.

**IoU** (Intersection over Union) measures overlap:

```
IoU = area(box1 ∩ box2) / area(box1 ∪ box2)
```

High overlap → duplicate detections on same block → NMS suppresses weaker box.

---

## 6. Second attribute: black marker (not from YOLO)

YOLO predicts **color class**. The repo adds **marker presence** with classical vision:

```python
marked = is_marked_with_black(frame[y1:y2, x1:x2])
color_label = f"{base_color}_marked" if marked else f"{base_color}_plain"
```

**Course alignment:** two attributes — **learned color** + **rule-based marker**.

---

## 7. Training context (conceptual)

`bestv8.pt` implies fine-tuning on labeled images of your blocks.

Typical workflow (not in main script):
1. Collect images in lab conditions.
2. Label boxes (Roboflow, CVAT, etc.).
3. Train YOLOv8: `yolo detect train data=... model=yolov8n.pt`
4. Export `best.pt` / `bestv8.pt`.

**Why custom weights?** Generic COCO YOLO knows “cup”, not your exact red cube under lab lights.

---

## 8. ASCII: YOLO in the stack

```
BGR frame
    │
    ▼
 YoloWorker thread
    │
    ▼
 list of {box, cx, cy, base_color, marked, color}
    │
    ▼
 filter inside_workspace()
    │
    ▼
 state machine → pick target
```

---

## 9. How this appears in the robotics repo

```python
model = YOLO(MODEL_PATH)  # bestv8.pt
yolo_worker = YoloWorker(model)
yolo_worker.start()
# each loop:
yolo_worker.push(frame)
valid_objects = [obj for obj in yolo_worker.results() if state.inside_workspace(...)]
```

**Note:** v5 does not filter by `box.conf` in code (threshold applied inside `model(..., conf=0.4)`). `workspace.py` exposes conf for debugging.

---

## 10. Common beginner confusion

1. **Class name must match `COLOR_TO_BIN_MAP` keys** — if YOLO says `Green` but map has `green`, bin lookup fails.
2. **Box center ≠ grasp point** — center is approximation; large errors near box edges.
3. **Latency** — inference ~50–150 ms; threaded worker hides most of this.
4. **GPU** — CPU works but slower; watch FPS.

---

## 11. Practice exercises

1. Run `model(frame)` and print all boxes with conf and class.
2. Draw boxes on image; change `YOLO_CONF` to 0.2 and 0.6 — compare false positives.
3. Verify each class name in `model.names` matches sorting dictionary.

---

## 12. Summary

YOLO provides **fast multi-object localization** with class labels. This repo uses it for **color**, then HSV for **marker**, then geometry for **where to move the arm**.

**Next:** [object_detection_pipeline.md](object_detection_pipeline.md)
