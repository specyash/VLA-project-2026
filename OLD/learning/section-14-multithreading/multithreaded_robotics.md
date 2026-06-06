# Multithreaded Robotics Pipelines

**Goal:** Understand why this project uses threads, how they communicate, and what can go wrong.

**Prerequisites:** [Section 1 — Python threading](../section-01-python-foundations/python_basics_for_cv.md)

**Next:** [real_time_robotics_architecture.md](real_time_robotics_architecture.md)

---

## 1. The timing problem

One thread doing everything:

```
read camera → YOLO (100ms) → ArUco → draw → robot move (15s) → repeat
```

During **robot move**, camera processing stops.  
During **YOLO**, UI stutters.

Robotics needs **parallel activities** with different deadlines.

---

## 2. Intuition: kitchen stations

| Station | Thread | Job |
|---------|--------|-----|
| Waiter | Main | Camera, UI, decisions, robot commands |
| Chef | YoloWorker | Heavy inference |
| Announcer | VoiceWorker | TTS queue |
| Analyst | VLMWorker | Remote LLM |

Orders (frames/text) go to queues; workers process when ready.

---

## 3. Threads in `automated_pipeline_v5.py`

### YoloWorker (`threading.Thread`, daemon=True)

- **Input:** `push(frame)` overwrites latest frame.
- **Output:** `results()` returns detection list.
- **Locks:** `_li` (input), `_lo` (output).

### VoiceWorker

- **Input:** `speak(text)` → `queue.put(text)`.
- **Output:** `pyttsx3` says text sequentially.
- **Shutdown:** `voice.queue.put(None)`.

### VLMWorker

- **Input:** `push_request(frame, prompt)` if not busy.
- **Output:** `latest_insight` string + voice announcement.
- **Shutdown:** `queue.put((None, None))`.

**Daemon=True:** process exit does not wait for thread forever — still send sentinels in `finally`.

---

## 4. Locks vs queues

| Mechanism | Used for |
|-----------|----------|
| `threading.Lock` | Protect `_frame`, `_out`, `PipelineState` tag updates |
| `queue.Queue` | Voice lines, VLM jobs (FIFO) |

**Rule:** Any variable touched by two threads needs a lock or a queue.

---

## 5. Producer–consumer pattern (YOLO)

```
Main:  push(frame) ──► [latest frame slot]
Worker: take frame ──► infer ──► write results
Main:  results() ◄── read list
```

**Latest-frame-only:** if inference is slow, intermediate frames drop — usually OK for tracking static cubes.

---

## 6. Global workers at import

```python
voice = VoiceWorker()
voice.start()
vlm_worker = VLMWorker()
vlm_worker.start()
```

Created when module loads — before `main()`.

**Implication:** importing the module starts TTS/VLM threads (side effect).

---

## 7. Race conditions to know

1. **`PipelineState`:** `update_tags` holds lock; `pixel_to_robot` may read homography without lock — rare torn update.
2. **Stale YOLO results:** results from frame N while display shows frame N+3.
3. **VLM `is_processing`:** drops new requests while busy — only one queued item anyway.

---

## 8. Why not asyncio here?

Arm SDK, OpenCV `wait_for_frames`, pyttsx3, and Ollama HTTP are **blocking**. Threads fit blocking I/O better than async for this codebase style.

---

## 9. How this appears in the robotics repo

| Component | Parallelism |
|-----------|-------------|
| Camera + UI | Main thread |
| YOLO | YoloWorker |
| Speech | VoiceWorker |
| VLM | VLMWorker |
| Robot moves | Main thread (blocking) |

**Bottleneck:** `execute_pick_and_place` on main thread — not parallelized.

---

## 10. Common beginner confusion

1. **More threads ≠ faster robot** — arm still one physical device.
2. **GIL** — Python threads still help when waiting on I/O or native libs (OpenCV, inference).
3. **copy() cost** — every `push` copies full frame — memory bandwidth matters.
4. **Daemon exit** — may cut speech mid-word on force quit.

---

## 11. Practice exercises

1. Print thread name in main vs `YoloWorker.run`.
2. Queue 5 `speak()` calls rapidly — observe sequential playback.
3. Measure FPS with YOLO thread on vs off (mock empty worker).

---

## 12. Summary

Multithreading keeps **perception and UI alive** while **slow jobs** (YOLO, TTS, VLM) run aside. Robot motion remains **serial** on the main thread in v5.

**Next:** [real_time_robotics_architecture.md](real_time_robotics_architecture.md)
