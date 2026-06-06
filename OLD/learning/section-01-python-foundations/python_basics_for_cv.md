# Python Foundations for Computer Vision & Robotics

**Goal:** Learn the Python patterns used everywhere in this repo before touching cameras or robots.

**Prerequisites:** None.

**Next:** [Section 2 — How images work](../section-02-images-and-cv-foundations/how_images_work.md)

---

## 1. Why Python for robotics vision?

Robotics labs use Python because:

- **Fast to experiment** — change detection thresholds in seconds
- **Rich libraries** — OpenCV, NumPy, YOLO, RealSense all have Python bindings
- **Readable** — vision pipelines are easier to debug when code reads like steps

This repo’s main script (`automated_pipeline_v5.py`) is ~650 lines of Python orchestrating camera, vision, and arm.

---

## 2. Intuition: a pipeline is a recipe

Think of the sorting robot like a kitchen:

```
Camera  →  "see ingredients"
Vision  →  "identify what's on the counter"
Brain   →  "decide which bin"
Arm     →  "pick and place"
```

Python **functions** are recipe steps. **Classes** bundle related steps and memory (e.g. “remember last calibration”). **Threads** let multiple recipes run at once (speak while detecting).

---

## 3. Syntax essentials

### Variables and types

```python
robot_ip = "192.168.1.152"   # str
confidence = 0.40            # float
tag_id = 16                  # int
connected = True             # bool
```

### Functions

```python
def pixel_to_robot(px, py):
    """Convert image pixel to robot mm — placeholder."""
    return px * 0.5, py * 0.5

x, y = pixel_to_robot(640, 360)
```

**Why functions?** Same math used in many places; fix once, fix everywhere.

### Classes

```python
class PipelineState:
    def __init__(self):
        self.homography = None

    def has_map(self):
        return self.homography is not None

state = PipelineState()
print(state.has_map())  # False
```

**Why classes?** `PipelineState` in the repo holds calibration, tags, and homography together with methods like `pixel_to_robot()`.

### Imports

```python
import cv2
import numpy as np
from ultralytics import YOLO
```

Imports load **packages** (folders of code). The repo never imports its own `workspace.py` — each script is standalone, but patterns repeat.

---

## 4. Lists and dictionaries

### Lists — ordered sequences

```python
corner_order = [16, 17, 18, 19]
home_pose = [0.0, -250.0, 300.0, -180.0, 0.0, -88.0]
```

### Dictionaries — key → value lookup

```python
color_to_bin = {
    "green": 24,
    "red": 20,
    "blue": 25,
    "yellow": 27,
}
bin_id = color_to_bin["green"]  # 24
```

**Robotics use:** Map detected **color name** → **bin AprilTag ID** (`COLOR_TO_BIN_MAP` in `automated_pipeline_v5.py`).

---

## 5. Loops

```python
for tag_id in [16, 17, 18, 19]:
    print(f"Looking for tag {tag_id}")

for i, color in enumerate(["red", "blue"]):
    print(i, color)  # 0 red, 1 blue
```

**While loop** — main camera loop:

```python
while True:
    ok, frame = cap.read()
    if not ok:
        break
    # process frame...
```

The repo’s `main()` uses `while True` until you press `q`.

---

## 6. NumPy basics

Images and geometry are **arrays of numbers**. NumPy makes that fast.

```python
import numpy as np

# 2D grid (like a tiny grayscale image)
img = np.zeros((100, 100), dtype=np.uint8)
img[50, 50] = 255  # bright dot at row 50, col 50

# Points for homography
pts = np.array([[100, 200], [300, 200]], dtype=np.float32)
```

**Shape convention:** `(height, width)` or `(rows, cols)` — **not** (x, y). This trips up beginners constantly.

```python
h, w = frame.shape[:2]  # color image: shape (h, w, 3)
```

---

## 7. Threading basics

**Problem:** YOLO inference takes ~50–100 ms. If the main loop waits, the UI freezes and you miss camera frames.

**Idea:** Run slow work on a **background thread**; main thread keeps reading the camera.

```python
import threading
import time

shared_result = []
lock = threading.Lock()

def worker():
    while True:
        time.sleep(0.1)
        with lock:
            shared_result.append("detection")

t = threading.Thread(target=worker, daemon=True)
t.start()
```

- **`daemon=True`** — thread exits when main program exits
- **`Lock`** — only one thread updates shared data at a time

---

## 8. Queues (producer–consumer)

A **queue** is a safe mailbox between threads.

```python
import queue
import threading

q = queue.Queue()

def producer():
    for i in range(5):
        q.put(f"frame_{i}")

def consumer():
    while True:
        item = q.get()  # blocks until something arrives
        print("got", item)
        q.task_done()

threading.Thread(target=consumer, daemon=True).start()
producer()
```

**Repo pattern:** `VoiceWorker` uses `queue.Queue()` for speech text; `VLMWorker` queues `(frame, prompt)` pairs.

```
Main thread          Voice thread
    |                    |
    speak("Hello") ----> queue
                         engine says it
```

---

## 9. Async vs threading (intuition)

| | **Threading** | **asyncio** |
|---|---------------|-------------|
| Best for | Blocking I/O (camera, robot SDK, TTS) | Many network waits |
| This repo | ✅ Used heavily | ❌ Not used |
| Mental model | Parallel workers | One worker, switch tasks while waiting |

Robotics code blocks on `arm.set_position(..., wait=True)` and `cap.read()` — **threads** are the practical choice here.

---

## 10. Producer–consumer in vision

```
┌─────────────┐   push(frame)   ┌─────────────┐
│ Main thread │ ──────────────> │ YoloWorker  │
│ (camera)    │ <────────────── │ (inference) │
└─────────────┘   results()     └─────────────┘
```

Main thread **produces** frames; YOLO thread **consumes** them and **produces** detections.

---

## How this appears in the robotics repo

| Concept | File / symbol |
|---------|----------------|
| Class holding state | `PipelineState`, `YoloWorker`, `VoiceWorker` |
| Dict mapping | `COLOR_TO_BIN_MAP`, `BIN_NAMES`, `TAG_TO_INDEX` |
| Thread + lock | `YoloWorker` (`_li`, `_lo` locks) |
| Queue | `VoiceWorker.queue`, `VLMWorker.queue` |
| NumPy arrays | Homography points, polygon `np.array` |
| Main loop | `main()` → `while True` |

---

## Common beginner confusion

1. **`import cv2` vs `pip install opencv-python`** — package name differs from import name.
2. **List index vs image coordinates** — image `(x,y)` is `(col, row)` = `(cx, cy)` from width/height.
3. **Thread safety** — never modify the same list from two threads without a `Lock`.
4. **`daemon` threads** — they won’t finish long tasks after `main()` exits; shutdown must be handled (`voice.queue.put(None)` in repo).

---

## Practice exercises

1. Write a dict mapping fruit names to bin letters; write a function `which_bin(fruit)` that returns `"unknown"` if missing.
2. Create a `Counter` class with `increment()` and `value`, used from two threads with a lock (increment 1000 times each — final value should be 2000).
3. Read `automated_pipeline_v5.py` lines 79–163 and list which classes use `queue.Queue`.

---

## Summary

- Python **functions** and **classes** structure the pipeline.
- **Dicts** encode sorting rules; **lists** encode poses and tag order.
- **NumPy** represents images and geometry.
- **Threads + queues** keep perception real-time while the arm moves slowly.

You are ready to learn what an **image** actually is in Section 2.
