# AI Models & AI Infrastructure — Full Reference

This single document covers **every AI / ML component** in the Autonomous Object Perception & Manipulation repository, how each is configured and wired, the infrastructure around them, and a focused **scope-for-improvement** section for the **model side** of the project.

> Scope note: this file is about the *intelligence* layer (detection model, vision-language model, speech, routing). Geometry (homography, ArUco), kinematics, and motion are covered in the section folders and are only referenced here where they touch the models.

---

## 1. Inventory of AI components

| # | Component | Type | Library | Where | Runs on |
|---|-----------|------|---------|-------|---------|
| 1 | **YOLOv8 (`bestv8.pt`)** | Object detector (CNN) | `ultralytics` | `automated_pipeline_v5.py` (`YoloWorker`) | Local (CPU/GPU) |
| 2 | **Llama 3.2 Vision** | Vision-Language Model (VLM) | `ollama` | `automated_pipeline_v5.py` (`VLMWorker`), `automated_pipeline_vlm.py`, `vlm_answering.py` | **Remote** Ollama server |
| 3 | **pyttsx3 TTS** | Speech synthesis (not learned at runtime) | `pyttsx3` | `VoiceWorker` | Local OS voices |
| 4 | **HSV black-marker classifier** | Classical rule “model” (2nd attribute) | `opencv` | `is_marked_with_black()` | Local |
| 5 | **VLM function-router (prototype)** | LLM tool-calling sketch | `requests` + Ollama | `vlm_answering.py` | Local Ollama |

Only **#1 and #2** are true neural models. **#3** is rule/voice synthesis. **#4** is deterministic CV but plays the role of a “classifier” for the second sorting attribute. **#5** is an unfinished routing experiment.

---

## 2. Model 1 — YOLOv8 detector (`bestv8.pt`)

### 2.1 Role
The **perception workhorse**. Detects colored cubes and returns bounding boxes + class (color) + confidence. This is the *required classical/ML baseline* detector for the sorting task.

### 2.2 Configuration (from `automated_pipeline_v5.py`)

```python
MODEL_PATH = "bestv8.pt"
YOLO_CONF  = 0.40       # confidence threshold
INF_INTERVAL = 0.08     # min seconds between inferences (~12.5 Hz cap)
# inside worker:
res = self.model(frame, conf=YOLO_CONF, iou=0.45, verbose=False)[0]
```

| Param | Value | Meaning |
|-------|-------|---------|
| weights | `bestv8.pt` | Custom-trained on this lab’s cubes |
| conf | 0.40 | Detections below 40% dropped |
| iou | 0.45 | NMS overlap threshold |
| interval | 0.08 s | Throttle to limit CPU/GPU load |

### 2.3 Output schema
Per detection the worker emits:

```python
{
  "box": (x1, y1, x2, y2),
  "cx": int, "cy": int,            # box center → fed to homography
  "base_color": "green",            # = model.names[class_id]
  "marked": True/False,             # from HSV classifier (#4)
  "color": "green_marked",          # combined attribute label
}
```

### 2.4 Inference infrastructure
- Wrapped in **`YoloWorker(threading.Thread)`** — async, latest-frame-only.
- Locks `_li` (input frame) / `_lo` (output list).
- Loaded **once** at startup: `model = YOLO(MODEL_PATH)`.
- No batching, no half-precision, no explicit device pinning (Ultralytics auto-selects).

### 2.5 Training (implied, not in repo)
`bestv8.pt` implies a fine-tune of a YOLOv8 base (likely `yolov8n/s`) on labeled cube images. **No training script, dataset, data.yaml, or class list is committed** — this is a gap (see §7).

---

## 3. Model 2 — Llama 3.2 Vision (VLM)

### 3.1 Role
**Optional, human-facing scene reasoning.** Produces natural-language descriptions of the workspace. It **never controls the arm** (course-compliant).

### 3.2 Configuration

`automated_pipeline_v5.py`:
```python
REMOTE_OLLAMA_IP = "192.168.239.57"
VLM_MODEL_NAME   = "llama3.2-vision"
self.client = ollama.Client(host=f'http://{REMOTE_OLLAMA_IP}:11434')
```

`automated_pipeline_vlm.py` uses the same model with a fixed “spatial scene graph” prompt and a **640×480** RealSense stream + matplotlib 3D plot.

### 3.3 Request path
```python
_, buffer = cv2.imencode('.jpg', frame)       # frame → JPEG bytes
img_str   = base64.b64encode(buffer).decode() # → base64
res = client.generate(model=VLM_MODEL_NAME, prompt=prompt, images=[img_str])
text = res['response']                          # then strip markdown for TTS
```

### 3.4 Prompts in use
- v5 CHAT key `1`: “Describe the spatial layout and the objects present…”
- v5 CHAT key `2`: “Identify the colors of the cubes and where they are located relative to each other.”
- `automated_pipeline_vlm.py`: one long “God Prompt” for full scene relationships, markdown-free.

### 3.5 Infrastructure
- **`VLMWorker(threading.Thread)`** — single in-flight request (`is_processing` guard).
- **Remote GPU server** over HTTP (`:11434`) — the laptop does not host the VLM.
- Output → `VoiceWorker` (spoken) + on-screen “Latest: …”.
- No retries, no timeout config, no streaming, no caching.

---

## 4. Model 3 — pyttsx3 Text-to-Speech

Not a learned model, but the **AI persona** layer.

```python
self.engine = pyttsx3.init()
# pick "zira"/female voice, rate 155, volume 0.9
```

- Runs in **`VoiceWorker`** via a `queue.Queue`.
- Offline, OS-dependent voices (SAPI5 on Windows).
- Sequential playback; long auto-sort runs can backlog phrases.

---

## 5. Model 4 — HSV black-marker classifier (second attribute)

The project’s **two attributes** = (color via YOLO) + (marker presence via this rule).

```python
def is_marked_with_black(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    black_mask = cv2.inRange(hsv, [0,0,0], [180,255,60])
    ratio = cv2.countNonZero(black_mask) / total_pixels
    return 0.02 < ratio < 0.40
```

- Deterministic, fast, explainable.
- Output sets `marked` and the `_marked/_plain` suffix.
- **Important:** detected but **not used** in auto-sort bin routing (see §7).

---

## 6. Model 5 — VLM function-router (`vlm_answering.py`, prototype)

A **tool-calling sketch**: VLM returns JSON `{"function_name", "args"}` routed to a Python registry (`analyze_stacking`, `check_clearance`, `locate_objects_on_markers`).

- Uses `requests.post("http://localhost:11434/api/generate")` with `format: "json"`, `temperature: 0.0`.
- **Incomplete** — file is truncated/broken (`else:` with no body) and **not imported** anywhere.
- Represents the intended path toward structured VLM-assisted decisions.

---

## 7. AI infrastructure overview

### 7.1 Topology
```
┌────────────────────── Laptop / Host PC ──────────────────────┐
│  automated_pipeline_v5.py (main process)                      │
│   ├─ YOLOv8 (bestv8.pt)   ← local inference (CPU/GPU)         │
│   ├─ HSV classifier        ← local OpenCV                      │
│   ├─ pyttsx3 TTS           ← local OS voices                   │
│   └─ VLMWorker ──HTTP──┐                                       │
└────────────────────────┼──────────────────────────────────────┘
                          │ base64 JPEG over :11434
                          ▼
        ┌───────────────────────────────────────┐
        │ Remote Ollama server 192.168.239.57    │
        │   llama3.2-vision (GPU)                 │
        └───────────────────────────────────────┘
```

### 7.2 Threading model for AI
| Model | Thread | Concurrency rule |
|-------|--------|------------------|
| YOLO | `YoloWorker` | latest-frame-only, ~12 Hz cap |
| VLM | `VLMWorker` | one request at a time |
| TTS | `VoiceWorker` | FIFO queue |

### 7.3 Dependencies (AI-relevant)
`ultralytics`, `ollama`, `pyttsx3`, `opencv(-contrib)-python`, `numpy`, (`pyrealsense2` for frames). **No pinned versions / requirements.txt** committed.

### 7.4 Model artifacts
- `bestv8.pt` — **binary weights, present but undocumented** (no dataset, no class map, no metrics).
- VLM weights — **not local**; live on the remote Ollama host.

---

## 8. Data flow through the AI layer

```
RGB frame ──► YOLOv8 ──► boxes + base_color + conf
                 │
         crop ROI│
                 ▼
          HSV black classifier ──► marked / plain
                 │
                 ▼
   combined label "color_marked"  ──► sort rule (color→bin)  ──► robot XY
                 
RGB frame ──► (CHAT only) VLM ──► text ──► TTS + overlay   (no robot path)
```

Key point: **YOLO + HSV feed control; VLM is a parallel narration branch.**

---

## 9. Scope for improvement — MODEL SIDE (the focus)

Ordered by impact. These are model/AI-layer changes only.

### P0 — Correctness & grading alignment

1. **Use confidence in logic.** `box.conf` is thresholded inside Ultralytics but never read or logged. Capture per-detection confidence, rank targets by it, and reject low-confidence picks. (Course explicitly asks for confidence estimation.)

2. **Use the second attribute in decisions.** `marked` is computed but ignored by `COLOR_TO_BIN_MAP`. Make the routing a function of **(color, marked)** so the dual-attribute requirement is actually exercised (e.g. marked → different bin or priority).

3. **Temporal stability / voting.** Single-frame YOLO drives picks. Add N-of-M frame agreement (or use Ultralytics **tracking** `model.track(..., persist=True)` with ByteTrack) so flicker doesn’t cause wrong picks or double-picks.

4. **Surface YOLO errors.** `except Exception: pass` in the worker hides inference failures. Log them; AI debugging is impossible otherwise.

### P1 — Detector quality & robustness

5. **Commit the training pipeline.** Add dataset, `data.yaml`, class list, training command, and metrics (mAP, per-class precision/recall). Without it `bestv8.pt` is a black box — the course penalizes black-box reliance.

6. **Validate class names vs `COLOR_TO_BIN_MAP` keys.** If `model.names` ever returns `Green`/`GREEN`, bin lookup silently fails. Normalize case and assert coverage at startup.

7. **Lighting/robustness.** Augment training (exposure, white balance, shadows) and re-evaluate `YOLO_CONF`/`iou` empirically rather than fixed 0.40/0.45.

8. **Right-size the model.** Pick `yolov8n` vs `s/m` based on measured FPS vs accuracy on the actual host; consider **FP16/TensorRT** export if a GPU is present, or **ONNX** for CPU speedups.

9. **Better grasp point than box center.** Box center is a weak grasp proxy. Consider segmentation (YOLOv8-seg) to grasp the **mask centroid**, improving pick reliability for non-centered or partially occluded cubes.

### P2 — VLM usage (within course rules)

10. **Make VLM useful, not decorative.** Use it for the **allowed** role: resolve ambiguous color/marker on a **cropped** object image (send the YOLO ROI, not the full 720p frame) and **compare** against the HSV/YOLO baseline in your report.

11. **Constrain VLM outputs.** Move toward the `vlm_answering.py` pattern: `format="json"`, `temperature=0.0`, a fixed schema, and validation — instead of free-form prose that must be string-parsed.

12. **Reliability controls.** Add timeout, retries/backoff, and a local fallback when `REMOTE_OLLAMA_IP` is unreachable. Currently a network failure just speaks an error.

13. **Reduce VLM payload cost.** Full-frame JPEG per query is heavy. Downscale/crop before `imencode`; cache last insight; debounce repeated requests.

14. **Finish or remove `vlm_answering.py`.** It’s broken (dangling `else:`) and unused. Either wire it as the structured router or delete to avoid confusion.

### P3 — Second-attribute classifier

15. **Make marker detection adaptive.** The HSV black thresholds (`V<60`, ratio 0.02–0.40) are hand-tuned and lighting-fragile. Calibrate per session (reuse `tune_hsv.py`) or replace with a tiny learned classifier on the ROI — and **estimate its confidence** too.

### P4 — MLOps / infrastructure hygiene

16. **`requirements.txt` with pinned versions** for `ultralytics`, `ollama`, `pyttsx3`, OpenCV, NumPy.
17. **Config file** for model path, thresholds, IPs (no hardcoded `192.168.*`).
18. **Metrics harness:** log detection FPS, confidence distribution, and end-to-end sort success rate over ≥10 objects (needed to prove the ≥80% target).
19. **Device/health check at startup:** confirm GPU/CPU, model load, and Ollama reachability; fail fast with a clear message.

---

## 10. Quick scorecard (model layer)

| Aspect | Current | Target |
|--------|---------|--------|
| Detector | YOLOv8 custom, works | + tracking, confidence, metrics |
| 2nd attribute | HSV rule, **unused in routing** | Used + confidence |
| Confidence handling | Threshold only | Logged + ranked + retried |
| VLM role | Decorative chat | Ambiguity resolution on crops, JSON-constrained |
| Reproducibility | `bestv8.pt` only | Dataset + train script + metrics + pins |
| Error visibility | `except: pass` | Logged + monitored |
| Robustness | Fixed thresholds | Tuned + augmented + adaptive |

---

## 11. One-paragraph summary

The project’s intelligence is a **local YOLOv8 detector** (`bestv8.pt`) for color/object detection, a **deterministic HSV rule** for the marker attribute, an **offline pyttsx3 voice**, and an **optional remote Llama 3.2 Vision VLM** (via Ollama) used only for narration. The infrastructure is a **monolithic threaded Python app** with async workers and one remote model dependency. The biggest model-side wins are not bigger models — they are **using confidence, using the marker attribute in routing, adding temporal stability, documenting/retraining the detector with metrics, and giving the VLM a constrained, course-legal job** (ambiguity resolution on cropped ROIs) instead of decorative chat.
