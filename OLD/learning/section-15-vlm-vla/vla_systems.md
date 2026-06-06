# Vision-Language-Action (VLA) Systems

**Goal:** Understand VLA as an embodied-AI concept, how it differs from this repo’s design, and where the course allows limited use.

**Prerequisites:** [vlm_fundamentals.md](vlm_fundamentals.md), [Section 16 — Full pipeline](../section-16-full-pipeline/full_pipeline_breakdown.md)

---

## 1. Intuition: from “what” to “do”

| Stage | Question answered |
|-------|-------------------|
| Vision | What is in the scene? |
| Language | What does the task mean in words? |
| **Action** | What should the robot do next? |

A **VLA model** maps **image + instruction** → **action** (joint deltas, gripper, waypoints).

Example instruction: *“Put the red block in the recycle bin.”*  
Desired output: motion command or skill invocation — not just text.

---

## 2. This repository is NOT a full VLA

```
Actual v5 architecture:

Image → YOLO + ArUco + homography  →  explicit rules  →  xArm API
Image → VLM (optional)             →  speech only
```

**No neural network outputs torques or Cartesian targets directly.**

That is intentional for course compliance and reliability.

---

## 3. Allowed VLA-style extensions (course)

You may add a VLA **only if**:

1. Classical pipeline works standalone (≥80% sort rate target).
2. VLA assists **decisions**, e.g.:
   - “Is this cube marked or plain?” when HSV uncertain.
   - Tie-break between two similar colors.
3. You **compare** VLA vs rule baseline in report.

**Disallowed:**
- Replacing homography with “model said where to go.”
- Prompt-only pick without geometric check.

---

## 4. Reference models (from course FAQ)

Names students may explore **offline** or for analysis:

| Model | Typical use in coursework |
|-------|----------------------------|
| **CLIP** | Image-text similarity for attributes |
| **LLaVA / BLIP** | Captioning, VQA on crops |
| **OpenVLA** | Research framework — high-level reasoning |
| **π0-style policies** | Simulation / analysis, not lab default |

None are required for a passing baseline in this repo.

---

## 5. Hypothetical VLA integration pattern (educational)

If you extended the project **without** breaking rules:

```
1. YOLO proposes crop
2. VLM/VLA answers: "marked=yes, color=red, confidence=high"
3. Python rule engine maps to bin_id 20
4. homography → (rx, ry)
5. execute_pick_and_place()
```

Action still executed by **explicit geometry** — VLA only labels.

---

## 6. Embodied AI vocabulary

| Term | Meaning |
|------|---------|
| **Embodied** | Agent has physical body in world |
| **Policy** | Function from observation → action |
| **End-to-end** | Raw pixels → motors with one network |
| **Modular pipeline** | This repo — separate perception, logic, control |

Course rewards **modular** clarity for integration grades.

---

## 7. ASCII: VLA vs modular

**End-to-end VLA (NOT this repo):**

```
Image + text ──► [single big model] ──► joint commands
```

**Modular (THIS repo):**

```
Image ──► YOLO/ArUco ──► state machine ──► xArm
Image ──► VLM ──► human speech (parallel)
```

---

## 8. How VLA appears in repo files

| File | VLA relevance |
|------|----------------|
| `automated_pipeline_v5.py` | CHAT mode only — not action |
| `vlm_answering.py` | Function-calling prototype |
| `automated_pipeline_vlm.py` | Spatial visualization + language |
| `project_guidelines.txt` | Policy on allowed use |

---

## 9. Risks of over-using VLA

- Non-repeatable demos.
- Hard to debug (“model felt like red”).
- Graders penalize black-box dependence.
- Latency breaks real-time sorting.

---

## 10. Common beginner confusion

1. **Voice output ≠ action** — speaking “I will pick red” is not VLA control.
2. **YOLO is not VLA** — it outputs classes, not language or skills.
3. **Remote Ollama** — still not VLA unless outputs drive verified motions.
4. **Bonus points** — need baseline comparison table in report.

---

## 11. Practice exercises

1. Write a table: which pipeline stages are geometric vs learned vs linguistic.
2. Propose one VLM-assisted attribute check that still uses homography for XY.
3. Read `project_guidelines.txt` section 5 — list three disallowed uses.

---

## 12. Summary

**VLA** = vision + language + **robot actions** in one policy.  
**This project** = vision + **explicit code** for actions, with optional VLM for **language only**.

Understanding the distinction explains project architecture choices and safe upgrade paths.

**Next:** [Full pipeline walkthrough](../section-16-full-pipeline/full_pipeline_breakdown.md)
