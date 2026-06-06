# Full Pipeline Walkthrough — Autonomous Object Perception & Manipulation

**Goal:** One end-to-end story of how every technology in this repository connects, with data types, coordinates, threads, and file references.

**Prerequisites:** All Sections 1–15 (especially 7–15).

**Primary code:** `automated_pipeline_v5.py` at repository root.

---

## Part A — The big picture (30 seconds)

You built a **modular perception–action sorter**:

1. **See** the table (RealSense or webcam).
2. **Know where the table is** (AprilTags + homography).
3. **Find cubes** (YOLO + optional black-marker HSV).
4. **Decide** which bin (color rules + state machine).
5. **Act** (xArm Cartesian moves + gripper).

Optional: **speak** (TTS) and **describe scene** (VLM) without controlling the arm.

```
┌──────────┐   ┌─────────────┐   ┌──────────┐   ┌─────────┐
│ Sensors  │ → │ Perception  │ → │  Logic   │ → │  Robot  │
│ RGB-D    │   │ tags+YOLO   │   │ sort FSM │   │ xArm    │
└──────────┘   └─────────────┘   └──────────┘   └─────────┘
                     │                │
                     └────── HUD / voice / VLM (parallel)
```

---

## Part B — Hardware & software map

| Layer | Hardware / library | Role |
|-------|-------------------|------|
| Sensor | Intel RealSense D4xx | 1280×720 color + depth |
| Sensor fallback | USB webcam | RGB only |
| Vision | OpenCV | I/O, ArUco, homography, HSV, UI |
| Detection | Ultralytics YOLOv8 (`bestv8.pt`) | Colored cube boxes |
| Fiducials | AprilTag 36h11 | Workspace + bins |
| Robot | UFACTORY xArm Lite 6 | Pick-and-place |
| Gripper | HTTP device or GPIO | Open/close |
| Speech | pyttsx3 | Status narration |
| VLM | Ollama remote `llama3.2-vision` | CHAT mode only |

**Supporting repo assets (not imported by v5, but part of the story):**

| File / folder | Purpose |
|---------------|---------|
| `workspace_calibration.yaml` | Saved tag pixel positions |
| `Lite6_arm_pos/lite6_end_effector.yaml` | Source of `ROBOT_READINGS` |
| `Camera_Caliberation/` | Chessboard intrinsics |
| `Arm_to_object/` | Alternative 3D calibration + pick |
| `workspace.py`, `run_pipeline.py`, `v3`–`v4` | Earlier pipeline iterations |

---

## Part C — Startup sequence (`main()`)

### C.1 Global workers (module import time)

```python
voice = VoiceWorker(); voice.start()
vlm_worker = VLMWorker(); vlm_worker.start()
```

Threads already running before camera opens.

### C.2 Robot connect

```python
arm = XArmAPI(ROBOT_IP)
# enable, mode 0, home, open gripper
```

Failure → `arm = None` (dry-run vision demo).

### C.3 Models and state

```python
model = YOLO("bestv8.pt")
yolo_worker = YoloWorker(model); yolo_worker.start()
state = PipelineState()  # loads YAML fallback, builds detector
stream_obj, read_frame = get_video_stream()
```

### C.4 UI windows

`Main View` + `TECHY SENSOR HUD` (resizable).

---

## Part D — The main loop (every frame)

### D.1 Acquire

```python
ok, frame, depth_frame = read_frame()
```

| Output | Type | Shape / notes |
|--------|------|----------------|
| `frame` | `uint8` BGR | `(720, 1280, 3)` |
| `depth_frame` | `uint16` mm or `None` | aligned to color if RealSense |

**Data origin:** `pyrealsense2` pipeline + `rs.align(color)` OR `cv2.VideoCapture`.

---

### D.2 User input

```python
key = cv2.waitKey(1) & 0xFF
```

| Key | Effect |
|-----|--------|
| `q` | Quit |
| `s` | Save live tag pixels → `workspace_calibration.yaml` |
| `r` | Clear fallback calib + bin stack offsets |
| `1/2/3` | Menu → auto / custom / chat (in MENU) |
| `c` | Cancel sub-modes |

---

### D.3 Fiducial perception (always when frame valid)

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
state.update_tags(gray)
```

**Inside `update_tags`:**

```
corners, ids = detector.detectMarkers(gray)

For each ID:
  center (cx, cy) = mean of 4 corners

  ID in {16,17,18,19} → workspace corner pixels
  ID in {20,24,25,27} → bin center pixels

If 4 workspace IDs present:
  live_workspace_pixels = {...}
else:
  live_workspace_pixels = None

_compute_homography():
  src = pixel corners [16,17,18,19] in order
  dst = ROBOT_READINGS XY for each corner
  H = cv2.findHomography(src, dst)
```

**Coordinate spaces:**

| Space | Units | Example use |
|-------|-------|-------------|
| Image pixel | px | tag center, YOLO `cx,cy` |
| Robot base XY | mm | `set_position(x,y,...)` |
| Robot Z | mm | `Z_LIMIT`, `HOVER_Z` (not from homography) |

**If polygon missing:** wait loop, voice “Waiting for workspace tags”, skip sort logic.

---

### D.4 Object detection (async)

```python
yolo_worker.push(frame)  # copy to worker
objs = yolo_worker.results()  # latest list from previous inference
```

**Worker output per object:**

```python
{
  "box": (x1,y1,x2,y2),
  "cx": int, "cy": int,
  "base_color": "green",      # YOLO class name
  "marked": True/False,       # HSV black ratio in ROI
  "color": "green_marked",    # combined label
}
```

**Marker logic (`is_marked_with_black`):**

```
ROI = frame[y1:y2, x1:x2]
HSV → inRange black → ratio of black pixels
0.02 < ratio < 0.40 → marked
```

**Filter:**

```python
valid_objects = [o for o in objs if state.inside_workspace(o["cx"], o["cy"])]
```

Expanded polygon (+75 px margin) prevents edge false negatives.

---

### D.5 Depth HUD (optional visualization)

```python
if depth_frame is not None:
    hud = create_techy_hud(frame, depth_frame, valid_objects)
    cv2.imshow("TECHY SENSOR HUD", hud)
```

Per object: `depth_frame[cy, cx]` printed in mm — **not** fed to arm Z in v5.

---

### D.6 Task state machine

States: `MENU`, `AUTO_SORT`, `CUSTOM_PICK_COLOR`, `CUSTOM_PICK_BIN`, `CHAT`.

#### AUTO_SORT (autonomous sorting path)

```
target = first valid_object where:
  COLOR_TO_BIN_MAP[base_color] in active_bins

if target:
  (rx, ry) = pixel_to_robot(target.cx, target.cy)
  bin_id = COLOR_TO_BIN_MAP[base_color]
  (bx, by) = pixel_to_robot(bin_pixel[bin_id])
  execute_pick_and_place(arm, (rx,ry), (bx,by), bin_id, bin_offsets, ...)
  drain 10 frames; yolo_worker.clear_results()
else:
  empty_frames++
  if empty_frames > 20 → return MENU ("workspace clear")
```

**Sorting rule table:**

| YOLO `base_color` | Bin AprilTag ID | Bin name |
|-------------------|-----------------|----------|
| green | 24 | Wet |
| red | 20 | Dangerous |
| blue | 25 | Dry |
| yellow | 27 | Recycle |

**Note:** `marked` does not change bin in current code.

#### CUSTOM modes

Human picks exact `color` string then bin index from visible `active_bins`.

#### CHAT mode

```python
vlm_worker.push_request(frame, prompt)
# → remote Ollama → voice.speak(answer)
```

No robot motion.

---

### D.7 Coordinate transform detail (critical path)

```
Object pixel (cx, cy)
        │
        ▼
cv2.perspectiveTransform(·, H)
        │
        ▼
(rx, ry) mm in robot base

Bin pixel (bx_px, by_px) ──same H──► (bx, by) mm
```

**Assumptions:**
- Table is a plane.
- H accurate when tags correct.
- Z independent (fixed pick/place heights).

---

### D.8 Manipulation (`execute_pick_and_place`)

**Inputs:** `(rx, ry)`, `(bx, by)`, `bin_id`, mutable `bin_offsets`.

**Waypoint script:**

| Step | Position | Action after move |
|------|----------|-------------------|
| 1 | object @ HOVER_Z | — |
| 2 | object @ Z_LIMIT | close gripper |
| 3 | object @ HOVER_Z | voice: transporting |
| 4 | bin @ HOVER_Z | — |
| 5 | bin @ drop_z | open gripper |
| 6 | bin @ HOVER_Z | — |
| 7 | HOME XYZ | voice: returning |

`drop_z = min(Z_LIMIT + bin_offsets[bin_id], HOVER_Z - 10)`  
Then `bin_offsets[bin_id] += 25`.

**Failure:** any `move_robot` fail → `recover_robot()` (home, clear errors).

**Blocking:** entire sequence on **main thread**.

---

### D.9 Render

```python
draw_annotations(frame, ...)  # polygon, boxes, bins
draw_menu_overlay(frame, app_state, ...)
cv2.imshow("Main View", frame)
```

---

## Part E — Thread interaction diagram (one sort cycle)

```
TIME →
Main:    [frame][tags][push yolo][pick target][==== BLOCKED PICK ====][draw]
YOLO:         [infer frame N][infer N+1][infer N+2] ...
Voice:              [speak pick][speak transport][speak home]
VLM:                                                      (idle)
```

During BLOCKED PICK, camera still reads in drain loop only (`for _ in range(10): read_frame()`).

---

## Part F — Data type flow summary

```
depth_frame[cy,cx] ──► int mm        (HUD only)
(cx, cy)           ──► H ──► (rx, ry) float mm
base_color         ──► dict ──► bin_id int
bin_id             ──► bin_offsets ──► drop_z float mm
(rx,ry,drop_z)     ──► set_position (xArm)
```

---

## Part G — Alternative paths in the same repository

Understanding the **full repo** means knowing v5 is one integration choice:

| Approach | File | Mapping method | Depth use |
|----------|------|----------------|-----------|
| **Main** | `automated_pipeline_v5.py` | Homography + tag centers | HUD |
| Early | `run_pipeline.py` | Homography, 1 bin | Optional RGB |
| Vision only | `workspace.py` | Polygon lock, HSV+YOLO | Optional |
| 3D pick | `Arm_to_object/object_pick.py` | Pose + Kabsch YAML | RealSense |
| VLM demo | `automated_pipeline_vlm.py` | None for arm | 3D plot |

---

## Part H — Course requirements checklist (as implemented)

| Requirement | Status in v5 |
|-------------|--------------|
| Detect multiple objects | ✅ YOLO |
| Two attributes | ⚠️ Color sorts; marker shown not used in rules |
| Calibrated manipulator | ⚠️ Homography + taught corners; full intrinsics optional |
| Autonomous run | ⚠️ Press `1`; not zero-touch start |
| Confidence | ⚠️ Threshold only, not logged |
| Retry on failure | ❌ Recovery only |
| Top-down pick-place | ✅ |
| VLA optional | ✅ VLM chat only |

Use this table for project improvement planning.

---

## Part I — Typical failure modes (debugging guide)

| Observation | Check |
|-------------|-------|
| “Waiting for 4 tags” | Print IDs; lighting; fallback YAML |
| Arm misses XY | Reteach `ROBOT_READINGS`; press `s`; verify tag order |
| Wrong bin | Class name vs `COLOR_TO_BIN_MAP`; bin tag visible |
| Picks same cube twice | `clear_results`; stability filter (not implemented) |
| No depth HUD | Install `pyrealsense2`; USB3 |
| VLM silent | Ping `REMOTE_OLLAMA_IP`; model pulled on server |
| Dry-run but “success” | `arm is None` |

---

## Part J — Mental model for exams / demos

**One sentence:**  
Pixels become millimeters through AprilTag homography; YOLO finds cubes; Python rules pick a bin; xArm runs a fixed hover–grasp–place script; threads keep vision and speech alive.

**Three coordinate questions you must answer:**
1. Where is the object in the image? → `(cx, cy)`
2. Where is that on the robot table? → `(rx, ry) = H · (cx, cy)`
3. How low does the arm go? → `Z_LIMIT` + stack offset, **not** depth map (in v5)

---

## Part K — Suggested learning path through this repo

1. Run `workspace.py` — tags + YOLO only.  
2. Run `run_pipeline.py --dry-run` — one pick logic.  
3. Read `automated_pipeline_v5.py` with this document side-by-side.  
4. Run `Pos_on_april_tags.py` — understand `ROBOT_READINGS`.  
5. Optional: `Camera_caliberation.py` + `object_pick.py` for 3D path.  
6. Optional: CHAT mode with Ollama for VLM.

---

## Part L — Practice capstone

Without looking at code, draw on one page:

1. System block diagram (sensors → threads → arm).  
2. Coordinate spaces and one conversion example with fake numbers.  
3. AUTO_SORT decision tree for two cubes (red plain, green marked).  
4. List what happens on `recover_robot()`.

Then verify against `automated_pipeline_v5.py`.

---

## Summary

The repository implements a **classical explicit pipeline**: geometry and rules connect vision to manipulation; learning (YOLO) handles detection; language (VLM) is optional narration. The **data spine** is:

**RGB-D frame → tags (H) + YOLO (objects) → (rx,ry) + bin rule → waypoint script → xArm.**

Master that spine and every file in the repo has a clear place.

---

**You have completed the learning roadmap.** Revisit [Section 7](../section-07-homography/homography_and_workspace_mapping.md) if homography math needs reinforcement, or [Section 14](../section-14-multithreading/real_time_robotics_architecture.md) for concurrency details.
