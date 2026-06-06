# Vision-Language Models (VLM) Fundamentals

**Goal:** Understand what VLMs are, how this repo calls Ollama with images, and how that differs from YOLO sorting.

**Prerequisites:** [Section 14 — Real-time architecture](../section-14-multithreading/real_time_robotics_architecture.md)

**Next:** [vla_systems.md](vla_systems.md)

**Official references:**
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Llama 3.2 Vision (Meta)](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

---

## 1. Intuition: image + question → language answer

A **vision-language model** accepts:
- An **image** (what the camera sees)
- A **text prompt** (your question)

and produces **text** (description, reasoning, counts).

```
Photo of table + "What colors are the cubes?" → "There is a red cube near..."
```

Unlike YOLO, it is **general** — not limited to classes you trained.

---

## 2. YOLO vs VLM in this project

| | YOLO (`bestv8.pt`) | VLM (`llama3.2-vision`) |
|---|-------------------|-------------------------|
| Output | Boxes + fixed classes | Free-form text |
| Speed | ~10+ Hz possible | Seconds |
| Reliability for sorting | High (trained on your cubes) | Variable phrasing |
| Controls robot in v5? | Indirectly (via rules) | **No** |
| Course role | **Required baseline** | Optional overlay |

---

## 3. How images are sent to Ollama

```python
_, buffer = cv2.imencode('.jpg', frame)
img_str = base64.b64encode(buffer).decode('utf-8')

res = client.generate(
    model="llama3.2-vision",
    prompt="Describe the spatial layout...",
    images=[img_str]
)
text = res['response']
```

**Steps:**
1. Compress frame to JPEG in memory.
2. Base64-encode for JSON HTTP API.
3. Remote server runs multimodal model.
4. Return text; clean markdown for speech.

`REMOTE_OLLAMA_IP = "192.168.239.57"` — model runs on **another machine** with GPU.

---

## 4. `VLMWorker` thread design

```python
def push_request(self, frame, prompt):
    if not self.is_processing:
        self.queue.put((frame.copy(), prompt))
        voice.speak("Analyzing visual data.")
```

- Only accepts new job if not busy.
- Runs `client.generate` off main thread.
- Speaks answer via `VoiceWorker`.

**CHAT app state** keys:
- `1` — layout description prompt
- `2` — colors and relative positions prompt

---

## 5. Course policy alignment

From `project_guidelines.txt`:

**Allowed:**
- Attribute reasoning from crops
- Resolving ambiguity
- Comparing rules vs learned answers

**Not allowed:**
- End-to-end image → joint angles
- Replacing calibration / detection / grasp planning

**This repo is compliant:** VLM never calls `set_position`.

---

## 6. `automated_pipeline_vlm.py` (standalone demo)

Separate script with:
- Periodic or triggered VLM analysis
- 3D point cloud plot from depth (`get_spatial_points`)
- Matplotlib scatter — “spatial scene graph”

Shows how VLM + depth **visualization** combine; still not closed-loop control.

---

## 7. `vlm_answering.py` (sketch)

Defines **function registry** (`analyze_stacking`, `check_clearance`, …) for future tool-calling — **not wired** to v5.

Concept: VLM chooses a function name → Python executes structured action.

---

## 8. ASCII: VLM data flow

```
BGR frame
    │
    ▼
imencode JPG → base64
    │
    ▼
HTTP Ollama generate (remote GPU)
    │
    ▼
text response → clean → VoiceWorker + on-screen "Latest: ..."
```

No path to `pixel_to_robot` or `execute_pick_and_place`.

---

## 9. Limitations and pitfalls

- **Latency:** 2–30+ seconds.
- **Hallucination:** may invent objects not present.
- **Non-deterministic:** same scene, different words.
- **Network:** lab IP must reach Ollama host.
- **Privacy:** scene images leave local machine.

Do not use VLM as sole source of bin decisions for graded autonomy demo.

---

## 10. Common beginner confusion

1. **VLM ≠ VLA** — VLA also outputs actions (next file).
2. **Base64 size** — full 720p JPEG every query is heavy.
3. **Markdown in speech** — repo strips `*#_` for TTS clarity.
4. **Key conflict** — CHAT mode uses `1`/`2` like MENU; context is state-gated.

---

## 11. Practice exercises

1. Call Ollama with a still image of your table; compare answer to YOLO detections.
2. Log wall-clock time for one VLM request.
3. Write one prompt that asks only yes/no: “Is there a red cube?” — evaluate reliability.

---

## 12. Summary

VLMs add **semantic language** about the scene. In this project they are an **optional human-facing layer** on top of a **classical geometry + YOLO** sorting core.

**Next:** [vla_systems.md](vla_systems.md)
