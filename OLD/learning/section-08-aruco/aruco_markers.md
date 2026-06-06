# ArUco / AprilTag Markers

**Goal:** Understand fiducial markers, how OpenCV detects them, and why this project uses them for workspace and bins.

**Prerequisites:** [Section 7 — Homography](../section-07-homography/homography_and_workspace_mapping.md)

**Next:** [pose_estimation.md](pose_estimation.md)

**Official references:**
- [OpenCV ArUco module](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [AprilTag library (36h11 family)](https://april.eecs.umich.edu/software/apriltag.html)

---

## 1. The problem markers solve

A camera gives you **pixels**. A robot needs **millimeters on the table**.

You could manually click “this pixel = this robot point” every session — fragile and slow.

**Fiducial markers** are printed patterns with a **known ID** and **known geometry**. The camera recognizes them reliably, so the system always has anchor points in the scene.

**Analogy:** GPS landmarks vs. guessing your position from blurry photos.

---

## 2. What is ArUco? What is AprilTag?

| Term | Meaning |
|------|---------|
| **ArUco** | OpenCV’s module for detecting square binary markers |
| **AprilTag** | A popular marker family (bit patterns + IDs) |
| **DICT_APRILTAG_36h11** | Dictionary with 36-bit codes, Hamming distance 11 — used in this repo |

This project uses OpenCV’s ArUco API with the **AprilTag 36h11** dictionary:

```python
import cv2.aruco as aruco

dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, params)
```

**Why 36h11?** Good balance of ID count and robustness to noise/occlusion for tabletop setups.

---

## 3. What a marker looks like to the computer

A marker is a **black-and-white square** with a unique internal bit grid.

```
┌─────────────────┐
│ ████  ░░  ████  │  Outer black border
│ ░░  ████  ░░    │  Inner pattern = ID bits
│ ████  ░░  ████  │
└─────────────────┘
```

Detection steps (simplified):

1. Find quadrilateral candidates in the image (contour + geometry).
2. Warp candidate to a canonical square.
3. Read bits → decode **marker ID**.
4. Refine corner locations to sub-pixel accuracy.

Output per marker:
- **4 corner points** in image pixels
- **ID** (integer)

---

## 4. Marker IDs in this repository

From `automated_pipeline_v5.py`:

| ID | Role |
|----|------|
| **16, 17, 18, 19** | Workspace table corners (define sorting area) |
| **20** | Dangerous bin |
| **24** | Wet bin |
| **25** | Dry bin |
| **27** | Recycle bin |

Corner order for homography:

```python
CORNER_ORDER = [16, 17, 18, 19]
TAG_TO_INDEX = {16: 0, 17: 1, 18: 2, 19: 3}
```

**Physical setup tip:** Mount tags flat, fully visible, not glare-covered. Consistent placement every lab session.

---

## 5. Detection code walkthrough (repo)

```python
def make_aruco_detector():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, params)
```

Each frame (grayscale):

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
corners, ids, rejected = detector.detectMarkers(gray)
```

`PipelineState.update_tags()` processes results:

```python
for i, mid in enumerate(ids.flatten()):
    cx = int(corners[i][0][:, 0].mean())
    cy = int(corners[i][0][:, 1].mean())
    if mid_int in TAG_TO_INDEX:
        found_ws[mid_int] = (cx, cy)   # workspace corner
    elif mid_int in BIN_NAMES:
        found_bins[mid_int] = (cx, cy) # bin center
```

**Important:** The code uses the **center of the 4 corners**, not one corner, as the reference pixel for that tag.

---

## 6. Live vs fallback calibration

```
┌──────────────────────────────────────────────┐
│  All 4 workspace tags visible this frame?    │
│     YES → live_workspace_pixels + homography │
│     NO  → fallback from workspace_calibration.yaml │
└──────────────────────────────────────────────┘
```

- **Live:** Homography updates every frame tags are seen.
- **Fallback:** Last saved pixel positions (`s` key saves YAML).

Bin tags merge: fallback positions overwritten by live detections when visible.

**Why both?** Tags may be occluded by the arm or objects; fallback keeps the polygon roughly valid.

---

## 7. ASCII: markers in the data flow

```
Grayscale frame
      │
      ▼
 detectMarkers()
      │
      ├─► 4 corner IDs (16–19) ──► workspace polygon
      │                              │
      │                              ▼
      │                         findHomography()
      │                              │
      └─► bin IDs (20,24,25,27) ──► bin pixel centers
                                       │
Object (cx,cy) ──────────────────────┴──► pixel_to_robot() ──► (x_mm, y_mm)
```

---

## 8. Tuning `DetectorParameters`

OpenCV lets you tune sensitivity. Defaults work for many labs; if detection is flaky:

- **Lighting:** diffuse light, reduce motion blur.
- **Print quality:** matte paper, correct scale (not too small in image).
- **Exposure:** avoid blown-out white regions in markers.

`workspace.py` uses more aggressive adaptive threshold windows for difficult cameras — compare if you have detection issues.

---

## 9. How this appears in the robotics repo

| File | Role |
|------|------|
| `automated_pipeline_v5.py` | Main detection + homography |
| `workspace_calibration.yaml` | Saved pixel positions |
| `Lite6_arm_pos/Pos_on_april_tags.py` | Record robot pose at each corner tag |
| `lite6_end_effector.yaml` | Robot XY at tags → `ROBOT_READINGS` |
| `Arm_to_object/object_pick.py` | Alternative 3D pose + Kabsch calibration |

---

## 10. Common beginner confusion

1. **ArUco vs AprilTag** — Here they are the same family via OpenCV’s dictionary name.
2. **Tag center vs tag corner** — Repo uses **center** for mapping; pose estimation (next lesson) uses full 3D pose from corners.
3. **Wrong ID printed** — Code maps ID 24 to Wet bin; wrong print = wrong bin.
4. **Only 3 tags visible** — Homography disabled until all 4 workspace tags found (or fallback has 4 entries).

---

## 11. Practice exercises

1. Run detection on a single printed tag; print `ids` and corner coordinates.
2. Cover one workspace tag — observe LIVE vs FALLBACK overlay colors in the UI.
3. Press `s` with live tags — open `workspace_calibration.yaml` and verify stored pixels.

---

## 12. Summary

- Markers are **known landmarks** linking image pixels to real-world layout.
- This project uses them for **workspace bounds**, **homography**, and **bin locations**.
- Detection runs on **grayscale** every frame inside `PipelineState.update_tags()`.

Next: [pose_estimation.md](pose_estimation.md) — 3D position and orientation of markers.
