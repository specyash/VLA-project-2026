# Project Proposal: Interactive and Adaptive HRI via Vision-Language Models on xArm Lite 6

## 1. Project Vision
The objective of this project is to build a reliable, context-aware tabletop manipulation pipeline that integrates Vision-Language-Action (VLA) principles with local speech interfaces on the xArm Lite 6. Rather than claiming machine learning model novelty, this work focuses on a robust engineering implementation of interactive HRI: enabling a robotic arm to resolve semantic ambiguity, respond to verbal mid-task corrections, and organize workspaces dynamically.

---

## 2. Concrete Task Scenarios
We will demonstrate the system on a tabletop workspace populated with everyday objects (colored blocks, stationery, containers, and household items).

### Scenario A: Semantic Object Retrieval (Phase 1)
* **Goal**: Retrieve an object based on natural language queries referencing implicit attributes.
* **Example**: User says, *"Give me something to write with."* The robot identifies the pen among waste and electronics, calculates its coordinate, and hands it to the user.

### Scenario B: Clarification Dialogue for Ambiguity Resolution (Phase 2)
* **Goal**: Detect ambiguous commands and clarify user intent before execution.
* **Example**: User says, *"Pick up the bottle."* If there are two bottles (e.g., green and clear), the robot halts and speaks: *"I see a green bottle and a clear bottle. Which one should I pick?"* It waits for the user's verbal response and proceeds accordingly.

### Scenario C: Verbal Interrupt & Mid-Motion Trajectory Replanning (Phase 2)
* **Goal**: Dynamically alter the trajectory or target mid-execution upon verbal correction.
* **Example**: User says, *"Place the red block in the bin."* While the robot is moving, the user says, *"Stop, put it in the box instead."* The robot halts immediately, computes the new destination, and completes the placement.

### Scenario D: Semantic Workspace Organization (Phase 3)
* **Goal**: Organize a cluttered tabletop using high-level relational concepts.
* **Example**: User says, *"Clean up my desk."* The robot classifies items into semantic categories (e.g., tools to the tool tray, trash to the bin, electronics to the charging zone) and sorts them.

---

## 3. System Architecture & Control Flow
The system operates as a distributed perception-cognition-execution loop.

```
       [User Voice]                  [RGB-D Camera]
            │                              │
            ▼                              ▼
    ┌───────────────┐              ┌───────────────┐
    │  Whisper STT  │              │ OpenCV/ArUco  │ (Calibration/Depth)
    └───────┬───────┘              └───────┬───────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
                 ┌───────────────────┐
                 │ VLM (Qwen2-VL)    │ <─── Prompt Template & Image
                 └─────────┬─────────┘
                           │ (Generates Action Plan JSON)
                           ▼
                 ┌───────────────────┐
                 │ Pick-Place        │
                 │ Task Planner      │
                 └─────────┬─────────┘
                           │ (Translates to joint/linear movements)
                           ▼
                 ┌───────────────────┐
                 │ xArm Python SDK   │
                 └─────────┬─────────┘
                           │ (Commands & Joint Limits)
                           ▼
                  [xArm Lite 6 Robot]
                           │
                           ▼
               ┌───────────────────────┐
               │    Piper TTS Engine   │ ───> [Voice Feedback to User]
               └───────────────────────┘
```

### Data and Control Flow:
1. **Perception**: The RealSense D435 camera captures RGB and depth frames. OpenCV tracks ArUco markers to run a Kabsch calibration, maintaining an active 3D transform matrix from Camera space to Robot Base space.
2. **Cognition**: Whisper STT converts user voice input to text. The VLM (Qwen2-VL) receives the current workspace image, the text command, and the 2D bounding boxes. It outputs a structured JSON action plan (e.g., `{"action": "pick", "target_id": "pen_1", "destination": "bin"}`).
3. **Execution**: The Custom Pick-Place Planner maps the target's pixel coordinates to 3D base coordinates using the Kabsch matrix and depth frame. It constructs a trajectory plan with joint-space movements for long transitions (avoiding collisions) and Cartesian linear movements for vertical pick/drop sequences.
4. **HRI Loop**: Piper TTS generates speech files dynamically to prompt the user or state actions.

---

## 4. Technology Stack & Model Selection

* **Vision-Language Model (VLM)**: Qwen2-VL-7B-Instruct. Selected for its state-of-the-art spatial layout capabilities, permitting precise bounding-box detection and semantic coordinate grounding. Run locally via Ollama/llama.cpp.
* **Speech-to-Text (STT)**: Whisper (base/small). Deployed locally to convert microphone capture into clean text strings.
* **Text-to-Speech (TTS)**: Piper. Highly optimized local neural text-to-speech engine that runs in sub-100ms latency, enabling real-time verbal replies.
* **Robotic Control**: `xarm-python-sdk` communicating over TCP/IP to the Lite 6 controller box. Using Position Control Mode (Mode 0) to mix linear Cartesian paths and P2P joint movements.
* **Perception Engine**: OpenCV (ArUco tracking for calibration, YOLOv8/v10 for object detection), RealSense SDK (`pyrealsense2`).

---

## 5. Development Phases & Success Criteria

### Phase 1: Baseline Semantic Manipulation
* **Scope**: Static camera-robot calibration (Kabsch), VLM-guided target selection from static images, and collision-free pick-and-place.
* **Success Criteria**: 
  - >90% success rate in translating semantic language commands to target coordinate selection.
  - >85% physical pick-and-place success rate on 10 trials.
  - Zero self-collision or table-collision incidents using joint-space transitions.

### Phase 2: Dialogue & Adaptive Integration
* **Scope**: Whisper/Piper integration, interactive clarification loops for ambiguous commands, verbal interruptions, and mid-motion trajectory replanning.
* **Success Criteria**:
  - <2.0 seconds end-to-end voice loop latency (STT -> VLM -> TTS).
  - >90% accuracy in detecting ambiguous scenes and generating clarification questions.
  - >80% success rate in recovering and replanning trajectories on verbal interrupts.

### Phase 3: Workspace Organization & Robustness
* **Scope**: Abstract task sorting, multi-object coordinate memory, grasping verification, and ROS 2 package wrapping (optional for lab integration).
* **Success Criteria**:
  - >85% accuracy in grouping items by dynamic, abstract categories (e.g. "organize my desk").
  - System runs continuously without requiring hardware restarts.

---

## 6. System Requirements

### Hardware:
* **Robot**: UFACTORY xArm Lite 6 robotic arm + digital TGPIO gripper.
* **Camera**: Intel RealSense D435 RGB-D camera.
* **Compute**: Intel Core i7/AMD Ryzen 7, 32GB RAM, NVIDIA RTX 3080/4090 (or Apple M-series Max) with CUDA/Metal acceleration to maintain local VLM/Whisper latency under 1.5 seconds.
* **Mounting**: Standard overhead rigid mounting frame for camera-to-workspace calibration.

### Software:
* **Operating System**: Ubuntu 22.04 LTS or macOS (ARM64).
* **Development Environment**: Python 3.10+, PyTorch (with CUDA support).
* **Libraries**: `opencv-python`, `numpy`, `pyyaml`, `pyrealsense2`, `xarm-python-sdk`, `whisper`, `piper-tts`.

---

## 7. Timeline & Task Division

### 16-Week Schedule:
* **Weeks 1–3 (System Setup)**: Rigid camera mounting, ArUco tag placement, Kabsch calibration testing, and baseline pick-place programming.
* **Weeks 4–6 (Phase 1 Baseline)**: Qwen2-VL prompting, grounding targets to coordinates, and demonstrating static semantic retrieval.
* **Weeks 7–10 (Phase 2 dialogue)**: Whisper and Piper integration, state machine formulation for clarification loops, and dialogue testing.
* **Weeks 11–13 (Phase 3 Organization)**: Multi-object manipulation, abstract workspace sorting routines, and safety envelope verification.
* **Weeks 14–16 (Evaluation & Writing)**: Systematic data collection on success rates, ablation studies on P2P transitions, and paper drafting.

### Team Roles:
1. **Yash (Perception & Calibration)**: Bounding box tracking, RealSense depth alignment, coordinate mapping calibration (Kabsch matrix), and OpenCV pipelines.
2. **Shradul (Planning & Robot Control)**: `xArm` SDK interface, joint-space transition calculations, collision avoidance, and gripper control.
3. **Teammate 3 (VLM & Speech Integration)**: Deploys Qwen2-VL, writes prompts for JSON action generation, configures Whisper STT and Piper TTS, and builds the HRI dialogue state machine.
