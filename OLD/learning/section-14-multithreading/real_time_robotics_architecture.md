# Real-Time Robotics Architecture

**Goal:** See the full concurrent architecture of `automated_pipeline_v5.py` — what runs when, and what “real-time” means here.

**Prerequisites:** [multithreaded_robotics.md](multithreaded_robotics.md), [Section 11 — Detection pipeline](../section-11-yolo/object_detection_pipeline.md)

**Next:** [Section 15 — VLM](../section-15-vlm-vla/vlm_fundamentals.md)

---

## 1. What “real-time” means in student projects

**Hard real-time:** guaranteed deadlines (industrial safety).  
**Soft real-time:** usually responsive; occasional lag OK.

This pipeline is **soft real-time**:
- Camera ~30 FPS target.
- YOLO ~10–12 Hz effective (`INF_INTERVAL` 0.08s).
- Robot moves human-scale seconds.

---

## 2. Layered architecture diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAIN THREAD                               │
│  RealSense read → ArUco → push YOLO → filter → state machine      │
│  → draw UI → imshow → (blocking) robot pick sequence             │
└────────────┬──────────────────┬──────────────────┬──────────────┘
             │ push frame       │ speak()          │ push_request
             ▼                  ▼                  ▼
     ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
     │ YoloWorker   │   │ VoiceWorker  │   │ VLMWorker    │
     │ inference    │   │ pyttsx3      │   │ Ollama HTTP  │
     └──────────────┘   └──────────────┘   └──────────────┘
```

---

## 3. One frame timeline (normal operation)

| Time | Main | YoloWorker |
|------|------|------------|
| t0 | `read_frame()` | sleeping / inferring |
| t1 | `update_tags(gray)` | — |
| t2 | `push(frame)` | receives copy |
| t3 | `results()` (previous) | running model |
| t4 | draw overlays | finishes → updates `_out` |
| t5 | `waitKey(1)` | — |

**Perception latency:** up to one inference period + frame copy.

---

## 4. AUTO_SORT timeline (pick event)

| Phase | Main thread | Other threads |
|-------|-------------|---------------|
| Detect target | running loop | YOLO updating |
| `pixel_to_robot` | homography | — |
| `execute_pick_and_place` | **blocked** ~10–30s | voice may talk |
| Drain 10 frames | read only | YOLO still runs |
| `clear_results` | empty list | — |

During pick, **no new auto-sort** until function returns.

---

## 5. Shared state map

| State | Owner | Readers |
|-------|-------|---------|
| `PipelineState.homography` | Main (tags) | Main (pick) |
| `yolo_worker._out` | YoloWorker | Main |
| `bin_offsets` | Main | Main |
| `app_state` | Main | Main |
| `vlm_worker.latest_insight` | VLMWorker | Main (draw) |

---

## 6. Synchronization choices

**Good:**
- Locks on YOLO frame/results.
- Queue for voice (no lost partial strings).

**Could improve:**
- Lock homography reads during tag update.
- Double-buffer detections with frame ID.
- Robot command queue thread (decouple move from vision).

---

## 7. Failure modes in concurrent systems

| Symptom | Possible cause |
|---------|----------------|
| Pick wrong object | Stale YOLO + unstable `next()` order |
| UI frozen | Long `set_position(wait=True)` |
| Speech backlog | Many `speak()` during auto-sort |
| VLM ignores request | `is_processing` True |
| HUD lag | `create_techy_hud` every frame at full res |

---

## 8. Comparison to ROS-style architecture

ROS would split:

- `/camera` node
- `/detector` node
- `/planner` node
- `/arm_driver` node

This repo is **monolithic Python** — faster to prototype, harder to scale teams. Concepts map 1:1 to future ROS nodes.

---

## 9. How this appears in the robotics repo

Entry: `main()` in `automated_pipeline_v5.py`.

Shutdown order in `finally`:

```python
yolo_worker.stop()
voice.queue.put(None)
vlm_worker.queue.put((None, None))
stream_obj.stop()  # or cap.release()
arm.disconnect()
```

Clean shutdown prevents hung camera and zombie speech.

---

## 10. Design principles used

1. **Latest frame wins** for inference — favor freshness over processing every frame.
2. **Fail soft on vision** — missing tags → wait or fallback YAML.
3. **Fail recover on arm** — `recover_robot` to home.
4. **Human UI thread** — OpenCV must run on main thread on many platforms.

---

## 11. Common beginner confusion

1. **30 FPS camera ≠ 30 YOLO FPS** — inference throttled.
2. **Threads don’t make robot faster** — only overlap perception and speech.
3. **CHAT mode still runs camera loop** — VLM async, UI responsive.
4. **Import side effects** — voice starts before `main()`.

---

## 12. Practice exercises

1. Add timestamp prints in main and YoloWorker — measure lag.
2. Disable HUD — note CPU drop.
3. Sketch sequence diagram for one full sort cycle across threads.

---

## 13. Summary

Real-time architecture = **main loop orchestration** + **background workers** for slow tasks. Robot blocking moves are the main latency outlier. Understanding thread boundaries explains most “it picked the wrong thing” race stories.

**Next:** [vlm_fundamentals.md](../section-15-vlm-vla/vlm_fundamentals.md)
