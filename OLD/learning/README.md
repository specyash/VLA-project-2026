# Robotics Vision Pipeline — Learning Roadmap

This folder teaches the **Autonomous Object Perception & Manipulation** repository from zero robotics background to full pipeline understanding.

Read sections **in order**. Each file builds on the previous ones.

## How to use this roadmap

1. Read one markdown file at a time.
2. Run the small Python examples on your machine (Python 3.8+).
3. Open the linked repo files and find the same idea in real code.
4. Do the **Practice** exercises before moving on.

## Repository entry point

After Section 16, study:

- **`automated_pipeline_v5.py`** — integrated sorting pipeline (camera → vision → robot)
- **`workspace_calibration.yaml`** — saved tag pixel positions
- **`bestv8.pt`** — YOLO weights (you train or download separately)
- **`Lite6_arm_pos/lite6_end_effector.yaml`** — robot corner poses used for homography

## Section index

| Section | Folder | Topics |
|---------|--------|--------|
| 1 | [section-01-python-foundations](section-01-python-foundations/) | Python, NumPy, threading, queues |
| 2 | [section-02-images-and-cv-foundations](section-02-images-and-cv-foundations/) | Pixels, cameras, projection |
| 3 | [section-03-opencv-fundamentals](section-03-opencv-fundamentals/) | OpenCV read/draw/video/process |
| 4 | [section-04-color-spaces](section-04-color-spaces/) | HSV, masks, color sorting |
| 5 | [section-05-contours](section-05-contours/) | Contours, bounding boxes |
| 6 | [section-06-coordinate-systems](section-06-coordinate-systems/) | Frames, transforms |
| 7 | [section-07-homography](section-07-homography/) | Pixel → table mapping |
| 8 | [section-08-aruco](section-08-aruco/) | AprilTags, pose |
| 9 | [section-09-camera-calibration](section-09-camera-calibration/) | Intrinsics, distortion |
| 10 | [section-10-depth](section-10-depth/) | RealSense, point clouds |
| 11 | [section-11-yolo](section-11-yolo/) | Object detection |
| 12 | [section-12-robotics](section-12-robotics/) | FK, IK, motion planning |
| 13 | [section-13-xarm](section-13-xarm/) | xArm API |
| 14 | [section-14-multithreading](section-14-multithreading/) | Real-time architecture |
| 15 | [section-15-vlm-vla](section-15-vlm-vla/) | Vision-language models |
| 16 | [section-16-full-pipeline](section-16-full-pipeline/) | End-to-end repo breakdown |

## Final goal

You should be able to explain:

- How a camera frame becomes robot `(x, y, z)` commands
- Why homography, ArUco tags, YOLO, and threading each exist in the pipeline
- How data and coordinates flow through `automated_pipeline_v5.py`

## Depth note (Sections 8–16)

Sections **1–7** and **8–16** are written at the same teaching standard: intuition first, repo mapping, diagrams, exercises, and official doc links where relevant. Section 16 is the **master end-to-end walkthrough** — read it last.
