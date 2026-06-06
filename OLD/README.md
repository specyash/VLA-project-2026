# RPM_Project_Lite6

 # Continuous Interactive Sorting Pipeline

An advanced, real-time robotic pick-and-place sorting system utilizing computer vision (YOLOv8), ArUco marker tracking, depth perception, and asynchronous voice synthesis. Designed for use with an xArm robotic arm and an Intel RealSense camera, this pipeline enables dynamic, state-driven sorting of objects with an interactive Topographic Depth HUD and natural voice announcements.

## ✨ Features

*   **Real-Time Object Detection:** Uses a threaded YOLOv8 worker to detect colored blocks without bottlenecking the main camera feed.
*   **Dynamic Workspace & Bin Tracking:** Utilizes AprilTag (36h11) ArUco markers to define a dynamic quadrilateral workspace and target bin locations. Includes a calibration fallback system.
*   **Topographic Depth HUD:** Leverages RealSense depth data to project a live, color-mapped topographic view of the workspace, displaying tracked objects with real-time Z-depth calculations.
*   **Asynchronous Natural Voice Announcer:** Uses threaded Text-to-Speech (`pyttsx3`) to narrate robotic actions, menu changes, and errors in real-time without freezing robot motion or camera feeds.
*   **Intelligent Pick & Place:** Features dynamic drop-height calculation (memory offsets) to safely stack blocks and prevent collisions.
*   **Interactive State Machine:** Switch seamlessly between Auto-Sort, Custom Pick & Place, and a placeholder LLM Chat mode using a keyboard-driven UI.
*   **Automated Error Recovery:** Built-in robotic arm fault detection that safely resets the arm to a home position and clears errors.

---

## 🛠️ Hardware Requirements

*   **Robotic Arm:** UFACTORY xArm (Configured via `xarm-python-sdk`).
*   **Gripper:** Custom HTTP-controlled gripper (or standard xArm gripper fallback).
*   **Camera:** Intel RealSense Depth Camera (e.g., D415, D435) is highly recommended for the Depth HUD. Standard webcams are supported as a fallback (RGB only).
*   **Fiducial Markers:** AprilTag 36h11 markers (IDs 16, 17, 18, 19 for workspace corners; IDs 20, 24, 25, 27 for bins).

---

## 📦 Software Dependencies

Ensure you have Python 3.8+ installed. Install the required packages using `pip`:

```bash
pip install opencv-python opencv-contrib-python numpy pyyaml requests pyttsx3 ultralytics
