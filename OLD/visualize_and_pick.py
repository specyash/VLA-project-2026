#!/usr/bin/env python3
"""
Continuous Interactive Sorting Pipeline:
1. Detects objects (green, blue, yellow) and bins (ID 20, 24).
2. Prompts user to select the target color via keyboard (g, b, y).
3. Prompts user to select the destination bin via keyboard (1 for 20, 2 for 24).
4. Executes the pick and place sequence.
5. Returns to selection loop until 'q' is pressed to quit.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time
import argparse
import sys
import requests

from ultralytics import YOLO

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────
ROBOT_IP = "192.168.1.152"
GRIPPER_IP = "192.168.0.147"
MODEL_PATH = "best.pt"
YOLO_CONF = 0.40
INF_INTERVAL = 0.08
STABLE_NEEDED = 10

TAG_TO_INDEX = {16: 0, 17: 1, 18: 2, 19: 3}
CORNER_ORDER = [16, 17, 18, 19]

BIN_NAMES = {
    20: "A",
    25: "B",
    26: "C",
    27: "D"
}

ROBOT_READINGS = {
    0: (53.86880500, -355.66052200),
    1: (47.09763700, -173.95874000),
    2: (-283.01510600, -193.80304000),
    3: (-245.37525900, -371.09149200)
}

Z_LIMIT = 176.0      
HOVER_Z = 276.0      
HOME_POSE = [0.0, -250.0, 300.0, -180.0, 0.0, -88.0]  

ROBOT_R = -178.0
ROBOT_P = 0.0
ROBOT_Y = -88.0

def make_aruco_detector():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, params)

COLOR_RANGES = {
    "yellow": [((20,  70,  50), (34,  255, 255))],
    "green":  [((35,  40,  25), (95,  255, 255))], # Expanded Hue to 95 and lowered Value to 25
    "blue":   [((96,  50,  50), (145, 255, 255))],
}

COLOR_BGR = {
    "yellow": (0, 255, 255), 
    "dark green": (0, 150, 0),
    "light green": (144, 238, 144),
    "blue": (255, 100, 0), 
    "orange": (0, 165, 255),
    "PINK": (203, 192, 255)
   
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
    if scores[best] < 0.1:  
        return "unknown", round(scores[best], 3)
    return best, round(scores[best], 3)

class PipelineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detector = make_aruco_detector()
        self.workspace_pixels = {}
        self.bin_pixels = {}
        self.homography = None
        self.locked = False
        self.stable_buf = []

    def update_tags(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        found = {}
        bins = {}

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                c = corners[i][0]
                cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
                mid_int = int(mid)
                if mid_int in TAG_TO_INDEX:
                    found[mid_int] = (cx, cy)
                elif mid_int in [20, 25, 26,27]:
                    bins[mid_int] = (cx, cy)

        with self.lock:
            self.workspace_pixels = found
            self.bin_pixels = bins

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

    def pixel_to_robot(self, px, py):
        if self.homography is None:
            return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        robot_pt = cv2.perspectiveTransform(pt, self.homography)[0][0]
        return robot_pt[0], robot_pt[1]

    def inside_workspace(self, px, py, margin=75):
        if not self.locked:
            return False
        pts = []
        for tid in CORNER_ORDER:
            if tid in self.workspace_pixels:
                pts.append(self.workspace_pixels[tid])
        if len(pts) != 4:
            return False
            
        # Calculate centroid
        poly = np.array(pts, dtype=np.float32)
        M = cv2.moments(poly)
        if M['m00'] == 0: return False
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        
        # Expand polygon outward by margin pixels
        expanded_pts = []
        for pt in pts:
            dx, dy = pt[0] - cx, pt[1] - cy
            dist = np.hypot(dx, dy)
            if dist == 0: dist = 1 # Avoid division by zero
            scale = (dist + margin) / dist
            expanded_pts.append((cx + dx * scale, cy + dy * scale))
            
        poly_exp = np.array(expanded_pts, dtype=np.float32)
        dist = cv2.pointPolygonTest(poly_exp, (float(px), float(py)), False)
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
                # Add IOU threshold to prevent merging of adjacent 1x1 inch blocks
                res = self.model(frame, conf=YOLO_CONF, iou=0.45, verbose=False)[0]
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    # Crop to the center 50% to prevent table background from diluting the color score
                    h_box, w_box = y2 - y1, x2 - x1
                    cy1, cy2 = y1 + h_box // 4, y2 - h_box // 4
                    cx1, cx2 = x1 + w_box // 4, x2 - w_box // 4
                    roi = frame[cy1:cy2, cx1:cx2]
                    
                    color, ccf = classify_color(roi)
                    out.append({"box": (x1, y1, x2, y2), "color": color, "conf": conf, "cx": cx, "cy": cy})
            except Exception:
                pass

            with self._lo:
                self._out = out
            time.sleep(INF_INTERVAL)

def gripper_open(arm):
    url = f"http://{GRIPPER_IP}/gripper/0"
    try:
        print(f"Opening gripper via {url} ...")
        resp = requests.get(url, timeout=2)
        if resp.status_code != 200:
            print(f"Warning: gripper HTTP open returned {resp.status_code}")
        time.sleep(3)
        return
    except Exception as e:
        print(f"Gripper HTTP open failed: {e}")

    if not arm:
        return
    print("Falling back to SDK: Opening gripper...")
    code = arm.set_cgpio_digital(0, 0)
    if code != 0:
        print(f"Warning: Gripper open returned code {code}")
    time.sleep(1)

def gripper_close(arm):
    url = f"http://{GRIPPER_IP}/gripper/1"
    try:
        print(f"Closing gripper via {url} ...")
        resp = requests.get(url, timeout=2)
        if resp.status_code != 200:
            print(f"Warning: gripper HTTP close returned {resp.status_code}")
        time.sleep(3)
        return
    except Exception as e:
        print(f"Gripper HTTP close failed: {e}")

    if not arm:
        return
    print("Falling back to SDK: Closing gripper...")
    code = arm.set_cgpio_digital(0, 1)
    if code != 0:
        print(f"Warning: Gripper close returned code {code}")
    time.sleep(1)

def execute_pick_and_place(arm, obj_xy, bin_xy):
    print(f"\n--- Executing Pick and Place ---")
    rx, ry = obj_xy
    bx, by = bin_xy
    
    def recover():
        print("\n[RECOVERY] Error or collision detected! Attempting to recover...")
        if arm:
            arm.clean_warn()
            arm.clean_error()
            arm.motion_enable(True)
            arm.set_state(0)
            time.sleep(0.5)
            gripper_open(arm)
            print("[RECOVERY] Moving to Safe Home...")
            arm.set_position(*HOME_POSE, speed=60, wait=True)
        print("[RECOVERY] Ready to try again.\n")
        return False
        
    def move_robot(x, y, z, r, p, yw, speed=60):
        print(f"Moving to: x={x:.1f}, y={y:.1f}, z={z:.1f}")
        if arm:
            if arm.error_code != 0:
                print(f"Controller is in error state: {arm.error_code}")
                return False
            code = arm.set_position(x, y, z, r, p, yw, speed=speed, wait=True)
            if code != 0 or arm.error_code != 0:
                print(f"Warning: xArm returned code {code}, err {arm.error_code}")
                return False
        return True
    
    if not move_robot(rx, ry, HOVER_Z, ROBOT_R, ROBOT_P, ROBOT_Y): return recover()
    
    safe_z = max(Z_LIMIT, 176.0)
    if not move_robot(rx, ry, safe_z, ROBOT_R, ROBOT_P, ROBOT_Y, speed=40): return recover()
    
    gripper_close(arm)
    if arm and arm.error_code != 0: return recover()
    
    if not move_robot(rx, ry, HOVER_Z, ROBOT_R, ROBOT_P, ROBOT_Y): return recover()
    if not move_robot(bx, by, HOVER_Z, ROBOT_R, ROBOT_P, ROBOT_Y): return recover()
    
    gripper_open(arm)
    if arm and arm.error_code != 0: return recover()
    
    if not move_robot(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2], HOME_POSE[3], HOME_POSE[4], HOME_POSE[5]): return recover()
    
    print("Sequence completed.\n")

def main():
    arm = None
    if XArmAPI is None:
        print("xArm SDK not installed. Running in dry-run mode.")
    else:
        try:
            print(f"Connecting to xArm Lite6 at {ROBOT_IP}...")
            arm = XArmAPI(ROBOT_IP, do_not_open=False)
            arm.clean_warn()
            arm.clean_error()
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            print("Robot connected.")
            gripper_open(arm)
            if arm:
                arm.set_position(*HOME_POSE, wait=True)
        except Exception as e:
            print(f"Failed to connect to robot: {e}. Running in visual mode only.")
            arm = None

    print(f"Loading YOLO model: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    yolo_worker = YoloWorker(model)
    yolo_worker.start()
    
    state = PipelineState()

    cap = cv2.VideoCapture(0)
    if REALSENSE_AVAILABLE:
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
    
    ui_state = "SELECT_COLOR"
    selected_color = None

    try:
        while True:
            ok, frame = read_frame()
            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            state.update_tags(gray)

            if state.locked:
                pts = []
                for tid in CORNER_ORDER:
                    if tid in state.workspace_pixels:
                        pts.append(state.workspace_pixels[tid])
                if len(pts) == 4:
                    pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
                    for tid in CORNER_ORDER:
                        pt = state.workspace_pixels[tid]
                        cv2.circle(frame, pt, 5, (0, 255, 255), -1)
                        cv2.putText(frame, f"ID {tid}", (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
                yolo_worker.push(frame)
                objects = yolo_worker.results()
                
                valid_objects = []

                # Draw all cubes
                for obj in objects:
                    if state.inside_workspace(obj["cx"], obj["cy"]):
                        valid_objects.append(obj)
                        
                        cx, cy = obj["cx"], obj["cy"]
                        x1, y1, x2, y2 = obj["box"]
                        color_name = obj["color"]
                        bgr_color = COLOR_BGR.get(color_name, (255, 255, 255))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
                        cv2.putText(frame, color_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)

                # Draw all Bins
                for bid, bpix in state.bin_pixels.items():
                    bin_cx, bin_cy = bpix
                    cv2.circle(frame, (bin_cx, bin_cy), 8, (255, 0, 0), -1)
                    bin_name = BIN_NAMES.get(bid, str(bid))
                    cv2.putText(frame, f"Bin {bin_name}", (bin_cx + 15, bin_cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Determine target if selected
                target = None
                if selected_color:
                    for obj in valid_objects:
                        if obj["color"] == selected_color:
                            target = obj
                            break
                            
                if target:
                    cx, cy = target["cx"], target["cy"]
                    x1, y1, x2, y2 = target["box"]
                    bgr_color = COLOR_BGR.get(target["color"], (0, 255, 0))
                    # Highlight with thicker bounds and dot
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 4)
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(frame, "TARGET", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Dynamic UI Interactions
                detected_colors = set(obj["color"] for obj in valid_objects if obj["color"] != "unknown")
                detected_bins = sorted(list(state.bin_pixels.keys()))
                
                if ui_state == "SELECT_COLOR":
                    if not detected_colors:
                        cv2.putText(frame, "Waiting for Cubes... | 'q' to Quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        color_opts = " ".join([f"'{c[0]}'({c.capitalize()})" for c in detected_colors])
                        cv2.putText(frame, f"Select Cube: {color_opts} | 'q' to Quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                elif ui_state == "SELECT_BIN":
                    if not target:
                        cv2.putText(frame, f"WARNING: {selected_color} cube lost! Press 'c' to cancel.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif not detected_bins:
                        cv2.putText(frame, f"Target: {selected_color.upper()}. No bins detected! Press 'c' to cancel.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        bin_opts = " or ".join([f"'{i+1}'(Bin {BIN_NAMES.get(bid, str(bid))})" for i, bid in enumerate(detected_bins)])
                        cv2.putText(frame, f"Target: {selected_color.upper()}. Press {bin_opts} | 'c' to Cancel", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Continuous Multi-Color Sorting", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                if ui_state == "SELECT_COLOR":
                    for c in detected_colors:
                        if key == ord(c[0]):  # Press first letter of color
                            selected_color = c
                            ui_state = "SELECT_BIN"
                            break
                            
                elif ui_state == "SELECT_BIN":
                    if key == ord('c'):
                        ui_state = "SELECT_COLOR"
                        selected_color = None
                    else:
                        for i, bid in enumerate(detected_bins):
                            if key == ord(str(i + 1)):
                                if target and bid in state.bin_pixels:
                                    robot_xy = state.pixel_to_robot(target["cx"], target["cy"])
                                    bin_px = state.bin_pixels[bid]
                                    bin_xy = state.pixel_to_robot(bin_px[0], bin_px[1])
                                    
                                    if robot_xy and bin_xy:
                                        cv2.putText(frame, "EXECUTING...", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                        cv2.imshow("Continuous Multi-Color Sorting", frame)
                                        cv2.waitKey(1)
                                        execute_pick_and_place(arm, robot_xy, bin_xy)
                                else:
                                    print(f"Cannot pick: Missing {selected_color} object or Bin {bid}.")
                                
                                ui_state = "SELECT_COLOR"
                                selected_color = None
                                break

            else:
                cv2.putText(frame, "Waiting for 4 Workspace Tags...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Continuous Multi-Color Sorting", frame)
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
