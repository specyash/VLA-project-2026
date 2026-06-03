# Workflow

This project is built around a simple simulation-first data flow:

```text
Camera
  -> HSV detector
  -> workspace mapping
  -> task logic
  -> fake robot
```

## Phase 1 Scaffold

Phase 1 creates the project shape without implementing runtime behavior. The
purpose is to make the boundaries explicit before adding camera loops,
detectors, calibration, planning, or execution.

## Data Flow

1. A camera source provides frames from a webcam or video file.
2. A perception module converts frames into stable `DetectedObject` records.
3. Calibration maps image pixels into simulated table coordinates.
4. Task logic decides what object should be sorted and where it should go.
5. The planner produces a pick-and-place sequence.
6. The simulated robot adapter logs the sequence.

## Coordinate Flow

```text
image pixel center -> homography -> table millimeters -> planner waypoints
```

The planner should not depend on OpenCV. It should receive table coordinates
and return structured robot steps.

## Simulation Boundary

The first working milestone will use:

- OpenCV webcam or video input
- HSV and contour-based detection
- manual four-corner workspace calibration
- homography-based pixel-to-table mapping
- fake robot command logging

Future phases may add ArUco or AprilTag calibration, YOLOv8, RealSense, xArm,
and voice control behind the same interfaces.
