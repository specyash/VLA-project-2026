# Object Detection Pipeline (Integration)

**Goal:** Understand how detections flow from camera frame to pick decisions in `automated_pipeline_v5.py`.

**Prerequisites:** [yolo_fundamentals.md](yolo_fundamentals.md), [Section 8 — ArUco](../section-08-aruco/aruco_markers.md)

**Next:** [Section 12 — Robotics fundamentals](../section-12-robotics/robotics_fundamentals.md)

---

## 1. End-to-end detection story (one frame)

```
1. read_frame()           → BGR image
2. update_tags(gray)      → workspace polygon, bin pixels, homography
3. yolo_worker.push(frame) → hand off to inference thread
4. yolo_worker.results()  → latest detection list
5. inside_workspace()     → drop objects outside table
6. state machine          → choose target, pixel_to_robot(), pick
```

Perception and action are **coupled** but **not in the same thread** for YOLO.

---

## 2. The `YoloWorker` class (producer–consumer)

### Producer (main loop)

```python
yolo_worker.push(frame)  # stores frame.copy() under lock
```

Only the **latest** frame is kept if inference falls behind — acceptable for real-time tracking; may skip intermediate frames.

### Consumer (worker thread)

```python
while self._alive:
    # grab frame
    res = self.model(frame, conf=YOLO_CONF, iou=0.45, verbose=False)[0]
    # build list of dicts
    with self._lo:
        self._out = out
    time.sleep(INF_INTERVAL)  # ~80 ms minimum between inferences
```

### Reader (main loop)

```python
objs = yolo_worker.results()  # copy of list under lock
```

---

## 3. Detection object schema

Each entry:

```python
{
    "box": (x1, y1, x2, y2),
    "color": "green_plain" or "red_marked",  # combined label
    "cx": cx, "cy": cy,
    "base_color": "green",
    "marked": True/False,
}
```

Downstream code uses:
- `base_color` → `COLOR_TO_BIN_MAP` (auto-sort)
- `color` → user selection in custom mode
- `cx, cy` → homography → robot XY

---

## 4. Workspace filtering

```python
valid_objects = [
    obj for obj in yolo_worker.results()
    if state.inside_workspace(obj["cx"], obj["cy"])
]
```

`inside_workspace` expands the workspace polygon outward (`margin=75` px) so edge objects are still included.

**Why?** YOLO may detect blocks slightly outside the strict tag quad; margin reduces false rejects.

---

## 5. Auto-sort target selection

```python
target = next(
    (obj for obj in valid_objects
     if COLOR_TO_BIN_MAP.get(obj["base_color"]) in active_bins),
    None
)
```

Logic:
1. Iterate detections (order not guaranteed stable).
2. First object whose color has a mapped bin **and** that bin tag is visible (or in fallback calib).
3. Transform centers to robot coordinates → `execute_pick_and_place`.

**Gaps (for your awareness):**
- No confidence ranking.
- No multi-frame confirmation.
- `marked` not used in routing.
- `next()` order can jitter.

---

## 6. After a pick: clearing stale detections

```python
for _ in range(10):
    read_frame()  # drain camera / let scene settle
yolo_worker.clear_results()
```

Prevents immediately re-picking the same object before YOLO refreshes.

---

## 7. Custom pick mode (human in the loop)

| State | User action | System |
|-------|-------------|--------|
| `CUSTOM_PICK_COLOR` | Press `0`,`1`,… | Select `green_plain`, etc. |
| `CUSTOM_PICK_BIN` | Press `1`,`2`,… | Map to sorted `active_bins` keys |

Allows wrong-bin testing or manual overrides — not fully autonomous.

---

## 8. ASCII: threads and data

```
 MAIN THREAD                    YOLO THREAD
 ───────────                    ───────────
 read_frame()                        
      │                              
 push(frame) ──────────────────► inference
      │                               │
 results() ◄────────────────── store _out
      │                              
 inside_workspace                    
      │                              
 pixel_to_robot                      
      │                              
 execute_pick_and_place (blocks main)
```

During pick, **no new auto-sort decisions** until sequence finishes.

---

## 9. Comparison: `workspace.py` detection path

Older script uses:
- HSV `classify_color` inside worker
- Confidence `conf` and color confidence `clr_conf`
- FPS printed

v5 trades some interpretability for YOLO robustness on colored cubes.

---

## 10. How to improve (conceptual, no code here)

| Improvement | Benefit |
|-------------|---------|
| Require 3/5 frames same detection | Stability |
| Sort by confidence × area | Better target |
| Use `marked` in bin rule | Full dual-attribute sort |
| Crop ROI to workspace before YOLO | Speed |

---

## 11. Common beginner confusion

1. **results() is stale by 1 frame** — normal with async inference.
2. **Empty list ≠ no objects** — may be inference lag or all outside polygon.
3. **Bin not in active_bins** — tag 24 not visible and not in YAML fallback.
4. **clear_results** — temporary empty list right after pick.

---

## 12. Practice exercises

1. Log `len(valid_objects)` and class names each frame for 30 s.
2. Cover a bin tag — verify that color no longer auto-sorts.
3. Trace one object from box → cx,cy → rx,ry on paper.

---

## 13. Summary

The detection pipeline is **async YOLO + geometric workspace filter + rule-based bin mapping**. Understanding the dict schema and thread boundaries is essential to debug wrong picks.

**Next:** [robotics_fundamentals.md](../section-12-robotics/robotics_fundamentals.md)
