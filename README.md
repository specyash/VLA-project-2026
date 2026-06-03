# Autonomous Object Perception and Manipulation

This project is a clean, simulation-first rebuild of an autonomous sorting
pipeline. The first milestone is to understand and test the workflow before any
real robot, depth camera, detector model, or voice assistant is introduced.

## Goal

The baseline workflow is:

```text
webcam or video file -> HSV/contour detection -> workspace mapping -> task logic -> simulated robot
```

The project is organized so each stage can be implemented and tested
independently:

- `src/camera`: webcam and video-file frame sources.
- `src/perception`: object schemas, HSV detection, and stability tracking.
- `src/calibration`: manual workspace calibration and homography mapping.
- `src/planning`: pick-and-place plan generation.
- `src/robot`: robot adapter interfaces and the fake robot implementation.
- `src/app`: UI helpers and application state machine.
- `tests`: focused tests for calibration, planning, routing, and robot logging.
- `data`: generated local data such as saved workspace calibration.
- `docs`: workflow and architecture notes.

## Current Phase

Phase 1 only creates the scaffold, dependencies, and documentation. Runtime
camera, detection, calibration, planning, and simulated robot behavior will be
added in later phases.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

The application entry point will be implemented in Phase 2.

```powershell
python -m src.main
```

## Safety

There is no real robot motion in the first milestone. The default implementation
will use a simulated robot adapter that logs commands instead of moving
hardware.

## Not Included Yet

These are future extension points only:

- YOLOv8 detector
- RealSense source
- xArm robot adapter
- ArUco or AprilTag calibration
- Voice assistant or command layer
