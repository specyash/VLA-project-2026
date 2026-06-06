#!/usr/bin/env python3
"""
Continuous Interactive Sorting Pipeline v2:
- Uses 'best1.pt' YOLO model for object detection.
- Checks for black square markers ('marked' vs 'plain').
- Dynamic/Fallback Workspace: Tracks workspace tags live, but relies on a saved YAML calibration if they are occluded.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time
import argparse
import sys
import requests
import yaml
import os

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
MODEL_PATH = "bestv8.pt"
YOLO_CONF = 0.40
INF_INTERVAL = 0.08
CALIB_FILE = "workspace_calibration.yaml"

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

COLOR_BGR = {
    "yellow": (0, 255, 255), 
    "green": (0, 255, 0),
    "blue": (255, 100, 0), 
    "red": (0, 0, 255)
}

def make_aruco_detector():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, params)

def is_marked_with_black(roi):
    """
    Checks if a cropped region (ROI) contains a small black square.
    """
    if roi.size == 0:
        return False
        
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60]) 
    
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0: return False
    
    black_pixels = cv2.countNonZero(black_mask)
    ratio = black_pixels / total_pixels
    
    if 0.02 < ratio < 0.40:
        return True
    return False

class PipelineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.detector = make_aruco_detector()
        self.live_workspace_pixels = {}
        self.live_bin_pixels = {}
        
        calib = self.load_calibration()
        self.fallback_workspace_pixels = calib.get('workspace_pixels', {})
        self.fallback_bin_pixels = calib.get('bin_pixels', {})
        
        self.homography = None
        self._compute_homography()

    def load_calibration(self):
        if os.path.exists(CALIB_FILE):
            try:
                with open(CALIB_FILE, 'r') as f:
                    data = yaml.safe_load(f)
                    res = {}
                    if data and 'workspace_pixels' in data:
                        res['workspace_pixels'] = {int(k): tuple(v) for k, v in data['workspace_pixels'].items()}
                    if data and 'bin_pixels' in data:
                        res['bin_pixels'] = {int(k): tuple(v) for k, v in data['bin_pixels'].items()}
                    return res
            except Exception as e:
                print(f"Error loading calibration: {e}")
        return {}

    def save_calibration(self, workspace_px, bin_px):
        data_to_save = {}
        if workspace_px:
            data_to_save['workspace_pixels'] = {k: list(v) for k, v in workspace_px.items()}
        if bin_px:
            data_to_save['bin_pixels'] = {k: list(v) for k, v in bin_px.items()}
            
        try:
            with open(CALIB_FILE, 'w') as f:
                yaml.dump(data_to_save, f)
            print(f"Calibration saved to {CALIB_FILE}")
            if workspace_px:
                self.fallback_workspace_pixels = workspace_px.copy()
            if bin_px:
                self.fallback_bin_pixels = bin_px.copy()
        except Exception as e:
            print(f"Error saving calibration: {e}")

    def update_tags(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        found_ws = {}
        found_bins = {}

        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                c = corners[i][0]
                cx, cy = int(c[:, 0].mean()), int(c[:, 1].mean())
                mid_int = int(mid)
                if mid_int in TAG_TO_INDEX:
                    found_ws[mid_int] = (cx, cy)
                elif mid_int in BIN_NAMES.keys():
                    found_bins[mid_int] = (cx, cy)

        with self.lock:
            if len(found_ws) == 4:
                self.live_workspace_pixels = found_ws
            else:
                self.live_workspace_pixels = None
            
            self.live_bin_pixels = found_bins
            self._compute_homography()

    def get_active_ws_pixels(self):
        if self.live_workspace_pixels:
            return self.live_workspace_pixels
        return self.fallback_workspace_pixels
        
    def get_active_bins(self):
        # We prefer fallback bins if available, as they are "locked in" and immune to arm occlusion.
        # If a bin isn't in fallback (e.g. first run), we use live.
        active = self.fallback_bin_pixels.copy()
        for bid, pos in self.live_bin_pixels.items():
            if bid not in active:
                active[bid] = pos
        return active

    def get_active_polygon(self):
        active_px = self.get_active_ws_pixels()
        if not active_px or len(active_px) != 4:
            return None
        pts = [active_px[tid] for tid in CORNER_ORDER]
        return np.array(pts, dtype=np.float32)

    def _compute_homography(self):
        active_px = self.get_active_ws_pixels()
        if not active_px or len(active_px) != 4:
            self.homography = None
            return

        pts_src = []
        pts_dst = []
        for k in CORNER_ORDER:
            px, py = active_px[k]
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
        poly = self.get_active_polygon()
        if poly is None:
            return False
            
        M = cv2.moments(poly)
        if M['m00'] == 0: return False
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        
        expanded_pts = []
        for pt in poly:
            dx, dy = pt[0] - cx, pt[1] - cy
            dist = np.hypot(dx, dy)
            if dist == 0: dist = 1
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
                res = self.model(frame, conf=YOLO_CONF, iou=0.45, verbose=False)[0]
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    class_id = int(box.cls[0])
                    base_color = self.model.names[class_id]
                    
                    roi = frame[y1:y2, x1:x2]
                    marked = is_marked_with_black(roi)
                    
                    final_color = f"{base_color}_marked" if marked else f"{base_color}_plain"
                    
                    out.append({
                        "box": (x1, y1, x2, y2), 
                        "color": final_color, 
                        "conf": conf, 
                        "cx": cx, "cy": cy, 
                        "base_color": base_color, 
                        "marked": marked
                    })
            except Exception as e:
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
    
    # Go up, move to bin, then GO DOWN before dropping
    if not move_robot(rx, ry, HOVER_Z, ROBOT_R, ROBOT_P, ROBOT_Y): return recover()
    if not move_robot(bx, by, HOVER_Z, ROBOT_R, ROBOT_P, ROBOT_Y): return recover()
    if not move_robot(bx, by, safe_z, ROBOT_R, ROBOT_P, ROBOT_Y, speed=40): return recover()
    
    gripper_open(arm)
    if arm and arm.error_code != 0: return recover()
    
    # Go back up so we don't knock the bin over on the way home
    if not move_robot(bx, by, HOVER_Z, ROBOT_R, ROBOT_P, ROBOT_Y): return recover()
    
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
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Save fallback workspace and bins if 's' is pressed and tags are visible
            if key == ord('s') and state.live_workspace_pixels:
                state.save_calibration(state.live_workspace_pixels, state.live_bin_pixels)
            elif key == ord('r'):
                print("Recalibrating... Fallback erased.")
                state.fallback_workspace_pixels = {}
                state.fallback_bin_pixels = {}
                if os.path.exists(CALIB_FILE):
                    os.remove(CALIB_FILE)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            state.update_tags(gray)

            active_poly = state.get_active_polygon()
            if active_poly is not None:
                # Draw workspace
                pts_int = np.array(active_poly, np.int32).reshape((-1, 1, 2))
                
                if state.live_workspace_pixels is not None:
                    cv2.polylines(frame, [pts_int], True, (0, 255, 0), 2)
                    cv2.putText(frame, "LIVE Workspace (Press 's' to save fallback)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.polylines(frame, [pts_int], True, (0, 255, 255), 2)
                    cv2.putText(frame, "FALLBACK Workspace (Tags blocked)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                yolo_worker.push(frame)
                objects = yolo_worker.results()
                
                valid_objects = []

                # Draw all valid cubes
                for obj in objects:
                    if state.inside_workspace(obj["cx"], obj["cy"]):
                        valid_objects.append(obj)
                        
                        cx, cy = obj["cx"], obj["cy"]
                        x1, y1, x2, y2 = obj["box"]
                        color_name = obj["color"] # e.g. "green_marked"
                        
                        if obj["marked"]:
                            bgr_color = (0, 165, 255) # Orange border for marked
                        else:
                            bgr_color = COLOR_BGR.get(obj["base_color"], (255, 255, 255))
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)
                        cv2.putText(frame, color_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)

                # Draw all Bins
                active_bins = state.get_active_bins()
                for bid, bpix in active_bins.items():
                    bin_cx, bin_cy = bpix
                    cv2.circle(frame, (bin_cx, bin_cy), 8, (255, 0, 0), -1)
                    bin_name = BIN_NAMES.get(bid, str(bid))
                    cv2.putText(frame, f"Bin {bin_name}", (bin_cx + 15, bin_cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Highlight selected target
                target = None
                if selected_color:
                    for obj in valid_objects:
                        if obj["color"] == selected_color:
                            target = obj
                            break
                            
                if target:
                    cx, cy = target["cx"], target["cy"]
                    x1, y1, x2, y2 = target["box"]
                    # Highlight with thicker bounds and dot
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 0, 200), 4) # Purple target box
                    cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                    cv2.putText(frame, "TARGET", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

                # Dynamic UI Interactions
                detected_colors = sorted(list(set(obj["color"] for obj in valid_objects)))
                detected_bins = sorted(list(active_bins.keys()))
                
                if ui_state == "SELECT_COLOR":
                    if not detected_colors:
                        cv2.putText(frame, "Waiting for Cubes... | 'q' to Quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    else:
                        # Since multiple classes might start with 'g' (e.g. green_marked, green_plain), we use Numbers to select
                        color_opts = " ".join([f"[{i}] {c}" for i, c in enumerate(detected_colors)])
                        cv2.putText(frame, f"Select Cube: {color_opts} | 'q' to Quit", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                elif ui_state == "SELECT_BIN":
                    if not target:
                        cv2.putText(frame, f"WARNING: {selected_color} cube lost! Press 'c' to cancel.", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif not detected_bins:
                        cv2.putText(frame, f"Target: {selected_color}. No bins! Press 'c' to cancel.", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        bin_opts = " or ".join([f"[{i+1}] for Bin {BIN_NAMES.get(bid, str(bid))}" for i, bid in enumerate(detected_bins)])
                        cv2.putText(frame, f"Target: {selected_color}. Press {bin_opts} | 'c' Cancel", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                cv2.imshow("Continuous Multi-Color Sorting v2", frame)
                
                if ui_state == "SELECT_COLOR":
                    for i, c in enumerate(detected_colors):
                        if key == ord(str(i)):  # Press index number to select color category
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
                                if target and bid in active_bins:
                                    robot_xy = state.pixel_to_robot(target["cx"], target["cy"])
                                    bin_px = active_bins[bid]
                                    bin_xy = state.pixel_to_robot(bin_px[0], bin_px[1])
                                    
                                    if robot_xy and bin_xy:
                                        cv2.putText(frame, "EXECUTING...", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                        cv2.imshow("Continuous Multi-Color Sorting v2", frame)
                                        cv2.waitKey(1)
                                        execute_pick_and_place(arm, robot_xy, bin_xy)
                                else:
                                    print(f"Cannot pick: Missing {selected_color} object or Bin {bid}.")
                                
                                ui_state = "SELECT_COLOR"
                                selected_color = None
                                break

            else:
                cv2.putText(frame, "Waiting for 4 Workspace Tags...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Continuous Multi-Color Sorting v2", frame)

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
