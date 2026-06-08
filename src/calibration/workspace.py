"""
ArUco Workspace Tracker
=======================
Handles the detection of 4 ArUco markers to define a workspace polygon,
and tracks their stability to auto-lock the workspace.

This is a clean extraction of the ArUco/Workspace tracking logic from the OLD pipeline.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time

# ── Configuration & Constants ──────────────────────────────────────────────────
# The dictionary to use. The old code used AprilTag 36h11, but if you switched
# to standard ArUco markers, you can change this to aruco.DICT_4X4_50, etc.
ARUCO_DICT = aruco.DICT_APRILTAG_36h11

# ArUco IDs → workspace corner role
TAG_ROLE = {
    16: "top_left",
    17: "top_right",
    18: "bottom_right",
    19: "bottom_left",
}
CORNER_ORDER = [16, 17, 18, 19]   # polygon winding order (TL→TR→BR→BL)
TAG_TO_INDEX = {16: 0, 17: 1, 18: 2, 19: 3}

# ArUco IDs for Dynamic Bins
BIN_NAMES = {
    24: "Wet Bin",       # Green
    20: "Dangerous Bin", # Red
    25: "Dry Bin",       # Blue
    27: "Recycle Bin"    # Yellow
}

# Workspace calibration save file
CALIB_FILE = "workspace_calibration.yaml"

# Physical coordinates of the ArUco markers in the robot's coordinate system
# Used for homography (pixel -> robot translation)
ROBOT_READINGS = {
    0: (53.86880500, -355.66052200),
    1: (47.09763700, -173.95874000),
    2: (-283.01510600, -193.80304000),
    3: (-245.37525900, -371.09149200)
}

STABLE_NEEDED = 20  # frames all 4 tags visible before locking

CORNER_LINE_COLOR = {
    16: (0,   255, 255),   # TL — yellow
    17: (255, 200,   0),   # TR — cyan-blue
    18: (  0, 165, 255),   # BR — orange
    19: (255,   0, 255),   # BL — magenta
}
CORNER_LABEL = {16: "TL", 17: "TR", 18: "BR", 19: "BL"}
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ══════════════════════════════════════════════════════════════════════════════
#  ARUCO DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
def make_aruco_detector():
    """Return an ArucoDetector using the configured dictionary."""
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.03
    return aruco.ArucoDetector(dictionary, params)


# ══════════════════════════════════════════════════════════════════════════════
#  ARUCO WORKSPACE TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class ArucoWorkspaceTracker:
    def __init__(self):
        self._detector = make_aruco_detector()
        self._lock = threading.Lock()
        self._positions = {}
        self._bin_positions = {}
        self._visible = {tid: False for tid in TAG_ROLE}
        self._stable_buf = []
        self._locked = False
        self.homography = None

    def update(self, gray_frame):
        """Find markers in the grayscale frame and update stability tracking."""
        corners, ids, _ = self._detector.detectMarkers(gray_frame)
        found_ws = {}
        found_bins = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                mid_int = int(mid)
                c = corners[i][0]
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())
                
                if mid_int in TAG_ROLE:
                    found_ws[mid_int] = [cx, cy]
                elif mid_int in BIN_NAMES:
                    found_bins[mid_int] = [cx, cy]

        with self._lock:
            for tid in TAG_ROLE:
                self._visible[tid] = tid in found_ws
            for tid, pos in found_ws.items():
                self._positions[tid] = pos
            
            self._bin_positions = found_bins

            # Check stability (all 4 workspace tags visible)
            if len(found_ws) == 4:
                self._stable_buf.append(dict(found_ws))
                if len(self._stable_buf) > STABLE_NEEDED:
                    self._stable_buf.pop(0)
                if len(self._stable_buf) >= STABLE_NEEDED and not self._locked:
                    self._locked = True
                    self._compute_homography()
            else:
                self._stable_buf.clear()

    def _compute_homography(self):
        """Compute the perspective transform from camera pixels to robot coordinates."""
        if not self._locked or not all(tid in self._positions for tid in CORNER_ORDER):
            self.homography = None
            return
            
        pts_src = [[self._positions[k][0], self._positions[k][1]] for k in CORNER_ORDER]
        pts_dst = [[ROBOT_READINGS[TAG_TO_INDEX[k]][0], ROBOT_READINGS[TAG_TO_INDEX[k]][1]] for k in CORNER_ORDER]
        
        # Calculate the 3x3 Homography Matrix
        self.homography, _ = cv2.findHomography(np.array(pts_src, dtype=np.float32), 
                                                np.array(pts_dst, dtype=np.float32))

    def pixel_to_robot(self, px, py):
        """Convert a pixel coordinate (x,y) to a real-world robot coordinate."""
        with self._lock:
            if self.homography is None: 
                return None
            pt = np.array([[[px, py]]], dtype=np.float32)
            robot_pt = cv2.perspectiveTransform(pt, self.homography)[0][0]
            return robot_pt[0], robot_pt[1]

    def reset(self):
        """Reset the calibration lock."""
        with self._lock:
            self._stable_buf.clear()
            self._locked = False

    def is_locked(self):
        """Check if the workspace is fully calibrated and locked."""
        with self._lock:
            return self._locked

    def stable_count(self):
        """Get the current stability frames count (out of STABLE_NEEDED)."""
        with self._lock:
            return len(self._stable_buf)

    def get_positions(self):
        """Get the center points of the detected corners."""
        with self._lock:
            return dict(self._positions)

    def get_bin_positions(self):
        """Get the current center points of the detected dynamic bins."""
        with self._lock:
            return dict(self._bin_positions)

    def get_visibility(self):
        """Get visibility status for each corner."""
        with self._lock:
            return dict(self._visible)

    def get_polygon(self):
        """Get the polygon formed by the 4 corners in order, if all are tracked."""
        with self._lock:
            if not all(tid in self._positions for tid in CORNER_ORDER):
                return None
            return np.array([self._positions[tid] for tid in CORNER_ORDER], dtype=np.int32)

    def inside_workspace(self, pt):
        """Check if a given (x,y) point is strictly inside the 4-corner workspace."""
        poly = self.get_polygon()
        if poly is None:
            return False
        return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING AND MARKINGS
# ══════════════════════════════════════════════════════════════════════════════
def draw_dashed_line(frame, p1, p2, color, thickness=1, dash=12):
    """Draw a dashed line between p1 and p2."""
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    dist = np.hypot(x2 - x1, y2 - y1)
    n = max(int(dist / dash), 1)
    for i in range(n):
        if i % 2 == 0:
            sx = int(x1 + (x2 - x1) * i / n)
            sy = int(y1 + (y2 - y1) * i / n)
            ex = int(x1 + (x2 - x1) * (i + 1) / n)
            ey = int(y1 + (y2 - y1) * (i + 1) / n)
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
    return dist


def draw_workspace(frame, tracker):
    """Draw the workspace polygon, corners, and their labels."""
    poly = tracker.get_polygon()
    if poly is not None:
        ov = frame.copy()
        cv2.fillPoly(ov, [poly], (0, 255, 255))
        cv2.addWeighted(ov, 0.08, frame, 0.92, 0, frame)
        cv2.polylines(frame, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)

    positions = tracker.get_positions()
    visible = tracker.get_visibility()
    
    for tid in CORNER_ORDER:
        if tid not in positions:
            continue
        px, py = positions[tid]
        col = (0, 255, 0) if visible.get(tid) else (0, 170, 255)
        cv2.circle(frame, (px, py), 9, col, -1)
        cv2.putText(frame, f"ID{tid} {CORNER_LABEL[tid]}",
                    (px + 12, py - 8), FONT, 0.50, col, 2, cv2.LINE_AA)

    # Draw the dynamic bins
    bin_positions = tracker.get_bin_positions()
    for bid, pos in bin_positions.items():
        bx, by = pos
        cv2.circle(frame, (bx, by), 8, (255, 0, 0), -1)
        cv2.putText(frame, f"Bin {BIN_NAMES.get(bid, bid)}", 
                    (bx + 15, by + 15), FONT, 0.6, (255, 0, 0), 2, cv2.LINE_AA)


def draw_hud(frame, tracker, cam_fps):
    """Draw the heads-up display showing calibration status."""
    h, w = frame.shape[:2]
    locked = tracker.is_locked()
    stab = tracker.stable_count()
    vis = tracker.get_visibility()
    n_vis = sum(vis.values())

    if not locked:
        pct = stab / STABLE_NEEDED
        bar_w = int((w - 40) * pct)
        cv2.rectangle(frame, (20, h - 38), (w - 20, h - 12), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, h - 38), (20 + bar_w, h - 12), (0, 200, 80), -1)
        missing = [f"ID{t}" for t in CORNER_ORDER if not vis.get(t)]
        msg = (f"Calibrating {stab}/{STABLE_NEEDED} frames..."
               if n_vis == 4 else f"Waiting for: {', '.join(missing)}")
        cv2.putText(frame, msg, (30, h - 17), FONT, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
    else:
        mode = "LOCKED" if n_vis == 4 else "PARTIAL (LOCKED)"
        bar = (f"Workspace: {mode}   Tags:{n_vis}/4"
               f"   cam:{cam_fps:.0f}fps"
               f"   Q=quit  R=reset")
        cv2.rectangle(frame, (0, h - 38), (w, h), (0, 60, 0), -1)
        cv2.putText(frame, bar, (12, h - 12), FONT, 0.52,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # Top-right corner tag indicators
    for i, tid in enumerate(CORNER_ORDER):
        col = (0, 255, 0) if vis.get(tid) else (0, 0, 200)
        cv2.circle(frame, (w - 25 - i * 22, 22), 8, col,
                   -1 if vis.get(tid) else 2)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # A standalone testing block for the ArUco portion of the pipeline.
    print("Starting ArUco Workspace Tracker test...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = ArucoWorkspaceTracker()

    WIN = "ArUco Workspace Tracker  |  Q=quit  R=reset"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    print(f"Tracking IDs {CORNER_ORDER}.")
    print("Keys: Q=quit  R=reset calibration\n")

    fcnt = 0
    cam_fps = 0.0
    t_cam = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        fcnt += 1
        now = time.time()
        if now - t_cam >= 1.0:
            cam_fps = fcnt / (now - t_cam)
            fcnt, t_cam = 0, now

        # ArUco requires grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Update tracker
        tracker.update(gray)

        # Draw workspace and HUD
        draw_workspace(frame, tracker)
        draw_hud(frame, tracker, cam_fps)

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker.reset()
            print("Calibration reset -- re-scanning.")

    cap.release()
    cv2.destroyAllWindows()
    print("Test finished.")
