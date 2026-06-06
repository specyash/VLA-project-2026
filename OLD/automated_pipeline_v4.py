#!/usr/bin/env python3
"""
Continuous Interactive Sorting Pipeline v5: 
Includes State Machine, Memory Offsets, Real-Time Topographic Depth HUD,
and Asynchronous Natural Voice Announcements.
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import queue
import time
import yaml
import os
import requests
import pyttsx3  # <-- Added for voice synthesis

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

# ── 1. Global Configuration & Mappings ────────────────────────────────────────
ROBOT_IP = "192.168.1.152"
GRIPPER_IP = "192.168.0.147"
MODEL_PATH = "bestv8.pt"
YOLO_CONF = 0.40
INF_INTERVAL = 0.08
CALIB_FILE = "workspace_calibration.yaml"

TAG_TO_INDEX = {16: 0, 17: 1, 18: 2, 19: 3}
CORNER_ORDER = [16, 17, 18, 19]

BIN_NAMES = {
    24: "Wet Bin",       
    20: "Dangerous Bin", 
    25: "Dry Bin",       
    27: "Recycle Bin"    
}

COLOR_TO_BIN_MAP = {
    "green": 24,
    "red": 20,
    "blue": 25,
    "yellow": 27
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

ROBOT_R, ROBOT_P, ROBOT_Y = -178.0, 0.0, -88.0

COLOR_BGR = {
    "yellow": (0, 255, 255), "green": (0, 255, 0),
    "blue": (255, 100, 0), "red": (0, 0, 255)
}

# ── 2. Voice Announcer Module (Threaded) ──────────────────────────────────────

class VoiceWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.queue = queue.Queue()
        self.engine = pyttsx3.init()
        
        # Make the voice sound more natural
        voices = self.engine.getProperty('voices')
        # Tries to select a female voice if available (often sounds smoother in pyttsx3)
        for voice in voices:
            if "female" in voice.name.lower() or "zira" in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break
        
        self.engine.setProperty('rate', 155)  # Slightly slower for a calmer, natural tone
        self.engine.setProperty('volume', 0.9)

    def speak(self, text):
        """Adds text to the speech queue without blocking the main thread."""
        self.queue.put(text)
        print(f"[VOICE]: {text}")

    def run(self):
        while True:
            text = self.queue.get()
            if text is None: break
            self.engine.say(text)
            self.engine.runAndWait()
            self.queue.task_done()

# Initialize Voice globally so helper functions can access it
voice = VoiceWorker()
voice.start()

# ── 3. Vision & Perception Modules ────────────────────────────────────────────

def make_aruco_detector():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, params)

def is_marked_with_black(roi):
    if roi.size == 0: return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 60]) 
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    total_pixels = roi.shape[0] * roi.shape[1]
    if total_pixels == 0: return False
    ratio = cv2.countNonZero(black_mask) / total_pixels
    return 0.02 < ratio < 0.40

class PipelineState:
    # ... [Keeping PipelineState exactly the same as your original script] ...
    def __init__(self):
        self.lock = threading.Lock()
        self.detector = make_aruco_detector()
        self.live_workspace_pixels, self.live_bin_pixels = None, {}
        
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
        if workspace_px: data_to_save['workspace_pixels'] = {k: list(v) for k, v in workspace_px.items()}
        if bin_px: data_to_save['bin_pixels'] = {k: list(v) for k, v in bin_px.items()}
        try:
            with open(CALIB_FILE, 'w') as f: yaml.dump(data_to_save, f)
            voice.speak("Workspace calibration saved successfully.")
            if workspace_px: self.fallback_workspace_pixels = workspace_px.copy()
            if bin_px: self.fallback_bin_pixels = bin_px.copy()
        except Exception as e:
            voice.speak("Warning. Failed to save calibration.")

    def update_tags(self, gray):
        corners, ids, _ = self.detector.detectMarkers(gray)
        found_ws, found_bins = {}, {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                cx, cy = int(corners[i][0][:, 0].mean()), int(corners[i][0][:, 1].mean())
                mid_int = int(mid)
                if mid_int in TAG_TO_INDEX: found_ws[mid_int] = (cx, cy)
                elif mid_int in BIN_NAMES.keys(): found_bins[mid_int] = (cx, cy)

        with self.lock:
            self.live_workspace_pixels = found_ws if len(found_ws) == 4 else None
            self.live_bin_pixels = found_bins
            self._compute_homography()

    def get_active_ws_pixels(self): return self.live_workspace_pixels or self.fallback_workspace_pixels
    
    def get_active_bins(self):
        active = self.fallback_bin_pixels.copy()
        for bid, pos in self.live_bin_pixels.items(): active[bid] = pos
        return active

    def get_active_polygon(self):
        active_px = self.get_active_ws_pixels()
        if not active_px or len(active_px) != 4: return None
        return np.array([active_px[tid] for tid in CORNER_ORDER], dtype=np.float32)

    def _compute_homography(self):
        active_px = self.get_active_ws_pixels()
        if not active_px or len(active_px) != 4:
            self.homography = None
            return
        pts_src = [[active_px[k][0], active_px[k][1]] for k in CORNER_ORDER]
        pts_dst = [[ROBOT_READINGS[TAG_TO_INDEX[k]][0], ROBOT_READINGS[TAG_TO_INDEX[k]][1]] for k in CORNER_ORDER]
        self.homography, _ = cv2.findHomography(np.array(pts_src), np.array(pts_dst))

    def pixel_to_robot(self, px, py):
        if self.homography is None: return None
        pt = np.array([[[px, py]]], dtype=np.float32)
        robot_pt = cv2.perspectiveTransform(pt, self.homography)[0][0]
        return robot_pt[0], robot_pt[1]

    def inside_workspace(self, px, py, margin=75):
        poly = self.get_active_polygon()
        if poly is None: return False
        M = cv2.moments(poly)
        if M['m00'] == 0: return False
        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        
        expanded_pts = []
        for pt in poly:
            dx, dy = pt[0] - cx, pt[1] - cy
            dist = max(np.hypot(dx, dy), 1)
            scale = (dist + margin) / dist
            expanded_pts.append((cx + dx * scale, cy + dy * scale))
            
        poly_exp = np.array(expanded_pts, dtype=np.float32)
        return cv2.pointPolygonTest(poly_exp, (float(px), float(py)), False) >= 0


class YoloWorker(threading.Thread):
    # ... [Keeping YoloWorker exactly the same as your original script] ...
    def __init__(self, model):
        super().__init__(daemon=True)
        self.model = model
        self._li, self._lo = threading.Lock(), threading.Lock()
        self._frame, self._out = None, []
        self._alive = True

    def push(self, frame):
        with self._li: self._frame = frame.copy()

    def results(self):
        with self._lo: return list(self._out)
    
    def clear_results(self):
        with self._lo: self._out = []

    def stop(self): self._alive = False

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
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    class_id = int(box.cls[0])
                    base_color = self.model.names[class_id]
                    
                    marked = is_marked_with_black(frame[y1:y2, x1:x2])
                    out.append({
                        "box": (x1, y1, x2, y2), 
                        "color": f"{base_color}_marked" if marked else f"{base_color}_plain", 
                        "cx": cx, "cy": cy, 
                        "base_color": base_color, "marked": marked
                    })
            except Exception: pass

            with self._lo: self._out = out
            time.sleep(INF_INTERVAL)


# ── 4. Robot Control Modules ──────────────────────────────────────────────────

def gripper_action(arm, state):
    """state: 0 for Open, 1 for Close"""
    action_str = "open" if state == 0 else "close"
    voice_str = "Releasing grip." if state == 0 else "Grasping object."
    voice.speak(voice_str)
    
    url = f"http://{GRIPPER_IP}/gripper/{state}"
    try:
        if requests.get(url, timeout=2).status_code != 200:
            voice.speak(f"Warning. Gripper {action_str} command failed.")
        time.sleep(3)
        return
    except Exception:
        print(f"Gripper HTTP {action_str} failed. Falling back to SDK.")

    if arm:
        arm.set_cgpio_digital(0, state)
        time.sleep(1)

def recover_robot(arm):
    voice.speak("Error detected. Initiating robot recovery sequence.")
    if arm:
        arm.clean_warn()
        arm.clean_error()
        arm.motion_enable(True)
        arm.set_state(0)
        time.sleep(0.5)
        gripper_action(arm, 0) # Open
        arm.set_position(*HOME_POSE, speed=60, wait=True)
        voice.speak("Recovery sequence complete. Returning to home position.")
    return False

def move_robot(arm, x, y, z, r, p, yw, speed=60):
    if arm:
        if arm.error_code != 0: return False
        if arm.set_position(x, y, z, r, p, yw, speed=speed, wait=True) != 0 or arm.error_code != 0: return False
    return True

def execute_pick_and_place(arm, obj_xy, bin_xy, bin_id, bin_offsets, item_desc="item"):
    bin_name = BIN_NAMES.get(bin_id, "the target bin")
    
    # Natural phrasing translation for item descriptions (e.g. "red_marked" -> "marked red block")
    if "_" in item_desc:
        parts = item_desc.split("_")
        spoken_item = f"{parts[1]} {parts[0]} block"
    else:
        spoken_item = f"{item_desc} block"

    voice.speak(f"Moving to pick up the {spoken_item}.")
    
    rx, ry = obj_xy
    bx, by = bin_xy
    safe_z = max(Z_LIMIT, 176.0) 
    
    current_offset = bin_offsets.get(bin_id, 0.0)
    drop_z = safe_z + current_offset
    drop_z = min(drop_z, HOVER_Z - 10.0) 
    
    steps = [
        (rx, ry, HOVER_Z, 60, None),         
        (rx, ry, safe_z, 40, lambda: gripper_action(arm, 1)), 
        (rx, ry, HOVER_Z, 60, lambda: voice.speak(f"Transporting to {bin_name}.")),         
        (bx, by, HOVER_Z, 60, None),         
        (bx, by, drop_z, 40, lambda: gripper_action(arm, 0)), 
        (bx, by, HOVER_Z, 60, None),         
        (*HOME_POSE[:3], 60, lambda: voice.speak("Returning to home position."))           
    ]
    
    for x, y, z, spd, action in steps:
        if not move_robot(arm, x, y, z, ROBOT_R, ROBOT_P, ROBOT_Y, speed=spd): 
            return recover_robot(arm)
        if action: action()
        
    bin_offsets[bin_id] = current_offset + 25.0
    print(f"Sequence completed. Bin {bin_name} next drop height increased to {safe_z + bin_offsets[bin_id]}mm.\n")


# ── 5. App Logic & Main UI ────────────────────────────────────────────────────

def create_techy_hud(color_img, depth_img, valid_objects):
    # ... [HUD code remains exactly the same] ...
    scale = 0.5
    h, w = int(depth_img.shape[0]*scale), int(depth_img.shape[1]*scale)
    depth_small = cv2.resize(depth_img, (w, h))
    color_small = cv2.resize(color_img, (w, h))
    depth_clipped = np.clip(depth_small, 100, 2500) 
    depth_norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_norm = 255 - depth_norm
    depth_norm[depth_small == 0] = 0 
    num_layers = 10
    layer_step = 256 // num_layers
    layered_depth = (depth_norm // layer_step) * layer_step
    depth_colormap = cv2.applyColorMap(layered_depth, cv2.COLORMAP_TURBO)
    depth_colormap[depth_small == 0] = [0, 0, 0] 
    edges = cv2.Canny(layered_depth, 50, 150)
    depth_colormap[edges > 0] = [0, 255, 0] 
    for obj in valid_objects:
        x1, y1, x2, y2 = [int(v * scale) for v in obj["box"]]
        cx, cy = int(obj["cx"] * scale), int(obj["cy"] * scale)
        color_name = obj["color"]
        obj_depth_mm = depth_img[obj["cy"], obj["cx"]]
        cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.circle(depth_colormap, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(depth_colormap, f"{color_name} ({obj_depth_mm}mm)", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(depth_colormap, "TOPOGRAPHIC DEPTH LAYERS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(depth_colormap, f"Tracked Objects: {len(valid_objects)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return np.hstack((color_small, depth_colormap))

def draw_annotations(frame, state, valid_objects, active_bins, target_color=None):
    # ... [Unchanged] ...
    poly = state.get_active_polygon()
    if poly is not None:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        color, text = ((0, 255, 0), "LIVE Workspace") if state.live_workspace_pixels else ((0, 255, 255), "FALLBACK Workspace")
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.putText(frame, f"{text} ('s' to save, 'r' to reset)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    for obj in valid_objects:
        x1, y1, x2, y2 = obj["box"]
        bgr = (0, 165, 255) if obj["marked"] else COLOR_BGR.get(obj["base_color"], (255, 255, 255))
        thickness = 4 if target_color and obj["color"] == target_color else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, thickness)
        cv2.putText(frame, obj["color"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
    for bid, (bx, by) in active_bins.items():
        cv2.circle(frame, (bx, by), 8, (255, 0, 0), -1)
        cv2.putText(frame, f"Bin {BIN_NAMES.get(bid, bid)}", (bx + 15, by + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def draw_menu_overlay(frame, app_state, detected_colors, detected_bins, selected_color=None):
    # ... [Unchanged] ...
    overlay_y = 60
    if app_state == "MENU":
        cv2.putText(frame, "[1] Auto-Sort All", (20, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "[2] Custom Pick & Place", (20, overlay_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "[3] Chat about Workspace", (20, overlay_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "[q] Quit", (20, overlay_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif app_state == "AUTO_SORT":
        cv2.putText(frame, "AUTO-SORTING... Press 'c' to Cancel.", (20, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    elif app_state == "CUSTOM_PICK_COLOR":
        if not detected_colors:
            cv2.putText(frame, "Waiting for Cubes... | 'c' to Cancel", (20, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            opts = " ".join([f"[{i}] {c}" for i, c in enumerate(detected_colors)])
            cv2.putText(frame, f"Select Cube: {opts} | 'c' Cancel", (20, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    elif app_state == "CUSTOM_PICK_BIN":
        bin_opts = " or ".join([f"[{i+1}] for {BIN_NAMES.get(b, b)}" for i, b in enumerate(detected_bins)])
        cv2.putText(frame, f"Target: {selected_color}. Press {bin_opts} | 'c' Cancel", (20, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    elif app_state == "CHAT":
        cv2.putText(frame, "LLM Chat Mode Active (See Terminal). Press 'c' to Exit.", (20, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 200), 2)

def get_video_stream():
    if REALSENSE_AVAILABLE:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        pipe.start(cfg)
        align = rs.align(rs.stream.color)
        def read_frame():
            frames = pipe.wait_for_frames()
            aligned = align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df: 
                return False, None, None
            return True, np.asanyarray(cf.get_data()), np.asanyarray(df.get_data())
        return pipe, read_frame
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        def read_frame():
            ok, fr = cap.read()
            return ok, fr, None
        return cap, read_frame

def main():
    voice.speak("System starting up. Initializing robot and vision pipeline.")
    arm = None
    if XArmAPI:
        try:
            arm = XArmAPI(ROBOT_IP, do_not_open=False)
            arm.clean_warn()
            arm.clean_error()
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            gripper_action(arm, 0)
            arm.set_position(*HOME_POSE, wait=True)
            voice.speak("Robotic arm connected successfully.")
        except Exception as e:
            voice.speak("Failed to connect to the robotic arm. Proceeding in dry run mode.")
            arm = None

    model = YOLO(MODEL_PATH)
    yolo_worker = YoloWorker(model)
    yolo_worker.start()
    state = PipelineState()
    stream_obj, read_frame = get_video_stream()

    app_state = "MENU"
    last_app_state = ""
    selected_color = None
    empty_frames = 0
    bin_offsets = {}
    missing_tags_announced = False
    
    cv2.namedWindow("TECHY SENSOR HUD", cv2.WINDOW_AUTOSIZE)
    voice.speak("Pipeline ready. Waiting for instructions from the main menu.")

    try:
        while True:
            ok, frame, depth_frame = read_frame()
            if not ok or frame is None: continue
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                voice.speak("Shutting down the system. Goodbye.")
                break
            if key == ord('s') and state.live_workspace_pixels: 
                state.save_calibration(state.live_workspace_pixels, state.live_bin_pixels)
            if key == ord('r'):
                voice.speak("Resetting workspace calibration.")
                state.fallback_workspace_pixels, state.fallback_bin_pixels = {}, {}
                bin_offsets = {}
                if os.path.exists(CALIB_FILE): os.remove(CALIB_FILE)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            state.update_tags(gray)

            if state.get_active_polygon() is None:
                if not missing_tags_announced:
                    voice.speak("Waiting for workspace tags. Please ensure all four corner tags are visible.")
                    missing_tags_announced = True
                
                cv2.putText(frame, "Waiting for 4 Workspace Tags...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Main View", frame)
                
                if depth_frame is not None:
                    hud = create_techy_hud(frame, depth_frame, [])
                    cv2.imshow("TECHY SENSOR HUD", hud)
                continue
            else:
                if missing_tags_announced:
                    voice.speak("Workspace mapped. Resuming normal operations.")
                    missing_tags_announced = False

            yolo_worker.push(frame)
            valid_objects = [obj for obj in yolo_worker.results() if state.inside_workspace(obj["cx"], obj["cy"])]
            active_bins = state.get_active_bins()

            if depth_frame is not None:
                hud = create_techy_hud(frame, depth_frame, valid_objects)
                cv2.imshow("TECHY SENSOR HUD", hud)

            # Announce state changes
            if app_state != last_app_state:
                if app_state == "MENU":
                    voice.speak("Returning to main menu.")
                elif app_state == "AUTO_SORT":
                    voice.speak("Auto sort mode engaged. Scanning for valid objects.")
                elif app_state == "CUSTOM_PICK_COLOR":
                    voice.speak("Custom pick and place engaged. Please select a target object.")
                elif app_state == "CHAT":
                    voice.speak("Chat mode engaged. Preparing context for the language model.")
                last_app_state = app_state

            # Process State Machine
            if app_state == "MENU":
                if key == ord('1'): app_state = "AUTO_SORT"
                elif key == ord('2'): app_state = "CUSTOM_PICK_COLOR"
                elif key == ord('3'): app_state = "CHAT"

            elif app_state == "AUTO_SORT":
                if key == ord('c'): 
                    app_state = "MENU"
                    empty_frames = 0
                else:
                    target = next((obj for obj in valid_objects if COLOR_TO_BIN_MAP.get(obj["base_color"]) in active_bins), None)
                    
                    if target:
                        empty_frames = 0 
                        bin_id = COLOR_TO_BIN_MAP[target["base_color"]]
                        rx, ry = state.pixel_to_robot(target["cx"], target["cy"])
                        bx, by = state.pixel_to_robot(active_bins[bin_id][0], active_bins[bin_id][1])
                        
                        cv2.putText(frame, f"AUTO-SORTING: {target['color']} -> {BIN_NAMES[bin_id]}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        cv2.imshow("Main View", frame)
                        cv2.waitKey(1) 

                        execute_pick_and_place(arm, (rx, ry), (bx, by), bin_id, bin_offsets, item_desc=target["color"])
                        
                        for _ in range(10): read_frame()
                        yolo_worker.clear_results() 
                        
                    else:
                        empty_frames += 1
                        if empty_frames > 20:  
                            voice.speak("Workspace clear. Auto sort routine finished.")
                            app_state = "MENU"
                            empty_frames = 0

            elif app_state == "CUSTOM_PICK_COLOR":
                detected_colors = sorted(list(set(o["color"] for o in valid_objects)))
                if key == ord('c'): app_state = "MENU"
                else:
                    for i, c in enumerate(detected_colors):
                        if key == ord(str(i)):
                            selected_color, app_state = c, "CUSTOM_PICK_BIN"
                            voice.speak(f"Selected {c.replace('_', ' ')}. Please select destination bin.")

            elif app_state == "CUSTOM_PICK_BIN":
                target = next((o for o in valid_objects if o["color"] == selected_color), None)
                detected_bins = sorted(list(active_bins.keys()))
                if key == ord('c'): app_state = "MENU"
                else:
                    for i, bid in enumerate(detected_bins):
                        if key == ord(str(i + 1)):
                            if target and bid in active_bins:
                                rx, ry = state.pixel_to_robot(target["cx"], target["cy"])
                                bx, by = state.pixel_to_robot(active_bins[bid][0], active_bins[bid][1])
                                cv2.putText(frame, "EXECUTING...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                cv2.imshow("Main View", frame)
                                cv2.waitKey(1)
                                execute_pick_and_place(arm, (rx, ry), (bx, by), bid, bin_offsets, item_desc=target["color"])
                                for _ in range(10): read_frame()
                                yolo_worker.clear_results()
                            app_state = "MENU"

            elif app_state == "CHAT":
                if key == ord('c'): 
                    app_state = "MENU"

            draw_annotations(frame, state, valid_objects, active_bins, selected_color if app_state == "CUSTOM_PICK_BIN" else None)
            draw_menu_overlay(frame, app_state, 
                              sorted(list(set(o["color"] for o in valid_objects))) if app_state == "CUSTOM_PICK_COLOR" else [], 
                              sorted(list(active_bins.keys())) if app_state == "CUSTOM_PICK_BIN" else [], 
                              selected_color)

            cv2.imshow("Main View", frame)

    finally:
        yolo_worker.stop()
        voice.queue.put(None) # Signal the voice thread to shut down
        if REALSENSE_AVAILABLE: stream_obj.stop()
        else: stream_obj.release()
        cv2.destroyAllWindows()
        if arm: arm.disconnect()

if __name__ == "__main__":
    main()