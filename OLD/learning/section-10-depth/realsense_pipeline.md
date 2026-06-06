# Intel RealSense Pipeline (pyrealsense2)

**Goal:** Understand how this repo starts the camera, aligns depth to color, and reads frames in the main loop.

**Prerequisites:** [depth_cameras.md](depth_cameras.md)

**Next:** [point_clouds.md](point_clouds.md)

**Official references:**
- [pyrealsense2 API](https://intelrealsense.github.io/librealsense/python_docs/_generated/pyrealsense2.html)
- [Align depth to other stream](https://github.com/IntelRealSense/librealsense/blob/master/doc/post-processing-filters.md)

---

## 1. Intuition: pipeline = assembly line for frames

The RealSense **pipeline** waits for synchronized sensor packets and delivers **framesets** (color + depth + optional IR).

You do not read `/dev/video0` directly — you use `rs.pipeline()`.

---

## 2. Configuration in this repo

From `get_video_stream()` in `automated_pipeline_v5.py`:

```python
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
pipe.start(cfg)
align = rs.align(rs.stream.color)
```

| Setting | Meaning |
|---------|---------|
| 1280×720 | Resolution (balance detail vs speed) |
| bgr8 | OpenCV-compatible color |
| z16 | 16-bit depth in mm |
| 30 FPS | Target frame rate |
| align(color) | Reproject depth into color pixel grid |

---

## 3. Reading one frame

```python
def read_frame():
    frames = pipe.wait_for_frames()      # blocking wait
    aligned = align.process(frames)    # depth matched to color pixels
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if not cf or not df:
        return False, None, None
    color = np.asanyarray(cf.get_data())
    depth = np.asanyarray(df.get_data())
    return True, color, depth
```

**Main loop:**

```python
ok, frame, depth_frame = read_frame()
```

- `frame` — BGR for vision and display.
- `depth_frame` — 2D array, same height/width as color after align.

---

## 4. Why alignment matters

Without alignment, `depth_img[cy, cx]` at the YOLO box center might sample the **wrong 3D point** (parallax error).

```
Camera baseline offset → depth and color rays differ
align.process()        → depth[i,j] matches color[i,j]
```

Always index depth with the **same** `(cx, cy)` as the color detection.

---

## 5. Fallback: OpenCV webcam

```python
cap = cv2.VideoCapture(0)
def read_frame():
    ok, fr = cap.read()
    return ok, fr, None
```

No depth → HUD skipped or empty; homography + YOLO still work.

---

## 6. Lifecycle and cleanup

```python
# startup
stream_obj, read_frame = get_video_stream()

# shutdown (finally block)
if REALSENSE_AVAILABLE:
    stream_obj.stop()
else:
    stream_obj.release()
```

Failing to `stop()` can leave the camera locked until unplug/reboot.

---

## 7. ASCII: RealSense in main loop

```
wait_for_frames()
       │
       ▼
 align to color
       │
   ┌───┴───┐
   ▼       ▼
 color   depth
   │       │
   ├─► ArUco / YOLO / draw
   └─► create_techy_hud (if depth not None)
```

---

## 8. Performance notes

- **1280×720 @ 30 FPS** + YOLO + dual OpenCV windows is heavy on CPU/GPU.
- HUD resizes to 0.5× — saves some cost.
- Blocking `wait_for_frames()` in main thread ties loop rate to camera FPS.

Possible improvements (for later coding): lower resolution for inference crop only; separate capture thread.

---

## 9. How this appears in the robotics repo

| Location | Behavior |
|----------|----------|
| `automated_pipeline_v5.py` | Primary RealSense + align |
| `run_pipeline.py` | Color-only RealSense option |
| `realsense_view.py` | Depth visualization experiments |
| `Camera_caliberation/Camera_caliberation.py` | Color stream for chessboard capture |
| `Arm_to_object/object_pick.py` | `RealSenseSource` class wrapper |

---

## 10. Common beginner confusion

1. **USB3 / power** — unstable streams if underpowered.
2. **Multiple apps** — only one process should own the pipeline.
3. **Depth scale** — use RealSense APIs if converting units; numpy array is usually mm for z16 in this setup.
4. **`REALSENSE_AVAILABLE` false** — package not installed; you are on webcam path silently.

---

## 11. Practice exercises

1. Print `frame.shape` and `depth_frame.shape` — confirm they match.
2. Overlay depth text at mouse position using `cv2.setMouseCallback`.
3. Measure loop FPS with and without HUD enabled.

---

## 12. Summary

`pyrealsense2` provides synchronized **color + depth** with **alignment** so detections and depth samples share pixel coordinates. The main pipeline feeds color to perception and depth to the topographic HUD.

**Next:** [point_clouds.md](point_clouds.md)
