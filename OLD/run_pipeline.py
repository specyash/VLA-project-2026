#!/usr/bin/env python3
"""
Full Sorting Pipeline:
1. Detect workspace using 4 AprilTags (16, 17, 18, 19).
2. Detect object using YOLO + Color classification.
3. Transform object camera pixels to robot XY using Homography.
4. Move robot above object, open gripper, move down (maintaining Z limit of 176mm), close gripper.
5. Find bin location (ArUco ID 20) using the same Homography.
6. Move to bin, open gripper, return home.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time
import argparse
import sys

from ultralytics import YOLO

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None
    print("xArm Python SDK not found. Install it with: pip install xarm-python-sdk")

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


# ── Configuration ─────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.1.152"
MODEL_PATH = "best.pt"
YOLO_CONF = 0.40
INF_INTERVAL = 0.08
STABLE_NEEDED = 20

# AprilTag IDs -> Robot reading index
TAG_TO_INDEX = {
    16: 0,
    17: 1,
    18: 2,
    19: 3
}

# From lite6_end_effector.yaml
ROBOT_READINGS = {
    0: (53.86880500, -355.66052200),
    1: (47.09763700, -173.95874000),
    2: (-283.01510600, -193.80304000),
    3: (-245.37525900, -371.09149200)
}

BIN_MARKER_ID = 20
Z_LIMIT = 176.0      # Picking height / table level
HOVER_Z = 276.0      # Hover height (Z_LIMIT + 100)
HOME_POSE = [0.0, -250.0, 300.0, -180.0, 0.0, -88.0]  # Safe home over workspace to avoid Error 22

ROBOT_R = -178.0
ROBOT_P = 0.0
ROBOT_Y = -88.0

# ──────────────────────────────────────────────────────────────────────────────

def make_aruco_detector():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, params)

COLOR_RANGES = {
    "green": [((35, 40, 40), (85, 255, 255))],
    # Add other colors if needed, but keeping it focused on the green cube
}

def classify_color(roi):
    if roi is None or roi.size == 0:
        return "unknown", 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total = hsv.shape[0] * hsv.shape[1]
    scores = {}
    for name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        scores[name] = np.count_nonzero(mask) / total
    if not scores: return "unknown", 0.0
    best = max(scores, key=scores.get)
    return best, round(scores[best], 3)

class PipelineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detector = make_aruco_detector()
        self.workspace_pixels = {}
        self.bin_pixel = None
        self.homography = None
        self.locked = False
        self.stable_buf = []

    def update_tags(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = {}
        bin_pos = None

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                c = corners[i][0]
                cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
                if int(mid) in TAG_TO_INDEX:
                    found[int(mid)] = (cx, cy)
                elif int(mid) == BIN_MARKER_ID:
                    bin_pos = (cx, cy)

        with self.lock:
            self.workspace_pixels = found
            self.bin_pixel = bin_pos

            if len(found) == 4:
                self.stable_buf.append(dict(found))
                if len(self.stable_buf) > STABLE_NEEDED:
                    self.stable_buf.pop(0)
                if len(self.stable_buf) >= STABLE_NEEDED and not self.locked:
                    self._compute_homography()
                    self.locked = True
            else:
                self.stable_buf.clear()

    def _compute_homography(self):
        pts_src = []
        pts_dst = []
        # Use the average of stable buffer
        avg_pixels = {k: [0, 0] for k in TAG_TO_INDEX}
        for state in self.stable_buf:
            for k, p in state.items():
                avg_pixels[k][0] += p[0]
                avg_pixels[k][1] += p[1]
        
        for k in TAG_TO_INDEX:
            px, py = avg_pixels[k][0] / STABLE_NEEDED, avg_pixels[k][1] / STABLE_NEEDED
            rx, ry = ROBOT_READINGS[TAG_TO_INDEX[k]]
            pts_src.append([px, py])
            pts_dst.append([rx, ry])
            
        self.homography, _ = cv2.findHomography(np.array(pts_src), np.array(pts_dst))
        print("[System] Workspace Locked. Homography computed.")

    def pixel_to_robot(self, px, py):
        if self.homography is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        robot_pt = cv2.perspectiveTransform(pt, self.homography)[0][0]
        return robot_pt[0], robot_pt[1]

    def inside_workspace(self, px, py):
        if not self.locked:
            return False
        pts = []
        for tid in [16, 17, 18, 19]:
            if tid in self.workspace_pixels:
                pts.append(self.workspace_pixels[tid])
        if len(pts) != 4:
            return False
        
        poly = np.array(pts, dtype=np.float32)
        dist = cv2.pointPolygonTest(poly, (float(px), float(py)), False)
        return dist >= 0

class YoloWorker(threading.Thread):
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model
        self._li = threading.Lock()
        self._lo = threading.Lock()
        self._frame = None
        self._out = []
        self._alive = True

    def push(self, frame):
        with self._li:
            self._frame = frame.copy()

    def results(self):
        with self._lo:
            return list(self._out)

    def stop(self):
        self._alive = False

    def run(self):
        while self._alive:
            frame = None
            with self._li:
                if self._frame is not None:
                    frame, self._frame = self._frame, None
            if frame is None:
                time.sleep(0.01)
                continue

            out = []
            try:
                res = self.model(frame, conf=YOLO_CONF, verbose=False)[0]
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    roi = frame[y1:y2, x1:x2]
                    color, ccf = classify_color(roi)
                    out.append({"box": (x1, y1, x2, y2), "color": color, "conf": conf, "cx": cx, "cy": cy})
            except Exception as e:
                pass

            with self._lo:
                self._out = out
            time.sleep(INF_INTERVAL)

def gripper_open(arm):
    print("Opening gripper...")
    arm.set_cgpio_digital(0, 0)
    time.sleep(1)

def gripper_close(arm):
    print("Closing gripper...")
    arm.set_cgpio_digital(0, 1)
    time.sleep(1)

def move_robot(arm, x, y, z, r, p, yw, speed=50):
    print(f"Moving to: x={x:.1f}, y={y:.1f}, z={z:.1f}")
    arm.set_position(x, y, z, r, p, yw, speed=speed, wait=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run without connecting to robot")
    parser.add_argument("--target-color", default="green", help="Target object color")
    args = parser.parse_args()

    arm = None
    if not args.dry_run:
        if XArmAPI is None:
            sys.exit("xArm SDK not installed.")
        print(f"Connecting to xArm Lite6 at {ROBOT_IP}...")
        arm = XArmAPI(ROBOT_IP)
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(True)
        arm.set_mode(0)
        arm.set_state(0)
        print("Robot connected.")
        gripper_open(arm)

    print(f"Loading YOLO model: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    yolo_worker = YoloWorker(model)
    yolo_worker.start()

    state = PipelineState()

    cap = cv2.VideoCapture(0)
    if REALSENSE_AVAILABLE:
        print("Using RealSense...")
        cap.release()
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        pipe.start(cfg)
        
        def read_frame():
            frames = pipe.wait_for_frames()
            cf = frames.get_color_frame()
            if not cf: return False, None
            return True, np.asanyarray(cf.get_data())
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        def read_frame():
            return cap.read()

    print("Pipeline ready. Waiting for workspace calibration (4 tags)...")

    task_completed = False
    
    try:
        while not task_completed:
            ok, frame = read_frame()
            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            state.update_tags(gray)

            if state.locked:
                yolo_worker.push(frame)
                objects = yolo_worker.results()
                
                target = None
                for obj in objects:
                    # Must be target color (or 'any') AND inside the workspace polygon
                    if (obj["color"] == args.target_color or args.target_color == "any"):
                        if state.inside_workspace(obj["cx"], obj["cy"]):
                            target = obj
                            break

                cv2.putText(frame, "Workspace Locked", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if target:
                    cx, cy = target["cx"], target["cy"]
                    cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"Target ({target['color']})", (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    robot_xy = state.pixel_to_robot(cx, cy)
                    if robot_xy:
                        rx, ry = robot_xy
                        cv2.putText(frame, f"Robot: {rx:.1f}, {ry:.1f}", (cx, cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        
                        # Proceed with the pick and place sequence if bin is also visible
                        if state.bin_pixel:
                            bin_cx, bin_cy = state.bin_pixel
                            bin_xy = state.pixel_to_robot(bin_cx, bin_cy)
                            cv2.circle(frame, (bin_cx, bin_cy), 8, (255, 0, 0), -1)
                            
                            print(f"\n--- Starting Pick and Place ---")
                            print(f"Target object at Robot X={rx:.1f}, Y={ry:.1f}")
                            print(f"Bin location at Robot X={bin_xy[0]:.1f}, Y={bin_xy[1]:.1f}")
                            
                            if not args.dry_run:
                                # 1. Move above object (Hover)
                                move_robot(arm, rx, ry, HOVER_Z, ROBOT_R, ROBOT_P, ROBOT_Y)
                                
                                # 2. Move down to object (Maintain Z limit of 176mm)
                                safe_z = max(Z_LIMIT, 176.0) # Ensure it never goes below 176mm
                                move_robot(arm, rx, ry, safe_z, ROBOT_R, ROBOT_P, ROBOT_Y)
                                
                                # 3. Close gripper
                                gripper_close(arm)
                                
                                # 4. Move up
                                move_robot(arm, rx, ry, HOVER_Z, ROBOT_R, ROBOT_P, ROBOT_Y)
                                
                                # 5. Move above bin
                                move_robot(arm, bin_xy[0], bin_xy[1], HOVER_Z, ROBOT_R, ROBOT_P, ROBOT_Y)
                                
                                # 6. Open gripper
                                gripper_open(arm)
                                
                                # 7. Move home
                                move_robot(arm, *HOME_POSE)
                                
                            task_completed = True
                        else:
                            cv2.putText(frame, "Waiting for Bin (ID 20)...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)

            else:
                cv2.putText(frame, f"Waiting for 4 Tags. Found: {len(state.workspace_pixels)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            cv2.imshow("Sorting Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        yolo_worker.stop()
        if not REALSENSE_AVAILABLE:
            cap.release()
        elif 'pipe' in locals():
            pipe.stop()
        cv2.destroyAllWindows()
        if arm:
            arm.disconnect()

if __name__ == "__main__":
    main()
