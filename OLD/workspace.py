"""
AprilTag Workspace Object Detector  —  RealSense D435 edition
=============================================================
Uses opencv-contrib-python ArUco detector (no C++ compiler needed).

Install:
    pip uninstall opencv-python -y
    pip install opencv-contrib-python pyrealsense2 ultralytics numpy

Tag family : DICT_APRILTAG_36h11
Corner IDs : 16=TL  17=TR  18=BR  19=BL

Workflow:
  1. Point camera at 4 tags  workspace auto-locks after 20 stable frames.
  2. YOLO detects objects inside polygon.
  3. HSV classifies object color.
  4. Dashed lines + pixel distances drawn from every corner tag to each object.

Keys:  Q = quit  |  R = reset calibration
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import time

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not found — falling back to OpenCV camera index 0.")

from ultralytics import YOLO

# ── User settings ─────────────────────────────────────────────────────────────
MODEL_PATH    = "best.pt"
YOLO_CONF     = 0.25
INF_INTERVAL  = 0.08          # seconds between YOLO calls (~12 fps detection)
STABLE_NEEDED = 20            # frames all 4 tags visible before locking
FALLBACK_CAM  = 0             # OpenCV camera index if RealSense unavailable

# AprilTag IDs → workspace corner role
TAG_ROLE = {
    16: "top_left",
    17: "top_right",
    18: "bottom_right",
    19: "bottom_left",
}
CORNER_ORDER = [16, 17, 18, 19]   # polygon winding order (TL→TR→BR→BL)

FONT = cv2.FONT_HERSHEY_SIMPLEX
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  APRILTAG DETECTOR  (opencv-contrib, no compiler needed)
# ══════════════════════════════════════════════════════════════════════════════
def make_aruco_detector():
    """Return an ArucoDetector using the AprilTag 36h11 dictionary."""
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    params     = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate    = 0.03
    return aruco.ArucoDetector(dictionary, params)


# ══════════════════════════════════════════════════════════════════════════════
#  HSV COLOR CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════
COLOR_RANGES = {
    "red":    [((0,   70,  50), (10,  255, 255)),
               ((170, 70,  50), (180, 255, 255))],
    "orange": [((11,  70,  50), (25,  255, 255))],
    "yellow": [((26,  70,  50), (34,  255, 255))],
    "green":  [((35,  40,  40), (85,  255, 255))],
    "blue":   [((86,  50,  50), (130, 255, 255))],
    "purple": [((131, 40,  40), (160, 255, 255))],
    "pink":   [((161, 40, 100), (169, 255, 255))],
    "white":  [((0,   0,  180), (180,  30, 255))],
    "gray":   [((0,   0,   60), (180,  30, 179))],
    "black":  [((0,   0,    0), (180,  30,  59))],
}
COLOR_BGR = {
    "red":    (0,   0,   220), "orange": (0,   140, 255),
    "yellow": (0,   220, 220), "green":  (0,   200,   0),
    "blue":   (220,  80,   0), "purple": (180,   0, 180),
    "pink":   (180, 100, 220), "white":  (230, 230, 230),
    "gray":   (150, 150, 150), "black":  ( 50,  50,  50),
}

def classify_color(roi):
    if roi is None or roi.size == 0:
        return "unknown", 0.0
    hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    total  = hsv.shape[0] * hsv.shape[1]
    scores = {}
    for name, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        scores[name] = np.count_nonzero(mask) / total
    best = max(scores, key=scores.get)
    return best, round(scores[best], 3)


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE TAG TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class TagTracker:
    def __init__(self):
        self._detector   = make_aruco_detector()
        self._lock       = threading.Lock()
        self._positions  = {}
        self._visible    = {tid: False for tid in TAG_ROLE}
        self._stable_buf = []
        self._locked     = False

    def update(self, gray):
        corners, ids, _ = self._detector.detectMarkers(gray)
        found = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                if int(mid) in TAG_ROLE:
                    c  = corners[i][0]
                    cx = int(c[:, 0].mean())
                    cy = int(c[:, 1].mean())
                    found[int(mid)] = [cx, cy]

        with self._lock:
            for tid in TAG_ROLE:
                self._visible[tid] = tid in found
            for tid, pos in found.items():
                self._positions[tid] = pos

            if len(found) == 4:
                self._stable_buf.append(dict(found))
                if len(self._stable_buf) > STABLE_NEEDED:
                    self._stable_buf.pop(0)
                if len(self._stable_buf) >= STABLE_NEEDED and not self._locked:
                    self._locked = True
            else:
                self._stable_buf.clear()

    def reset(self):
        with self._lock:
            self._stable_buf.clear()
            self._locked = False

    def is_locked(self):
        with self._lock:
            return self._locked

    def stable_count(self):
        with self._lock:
            return len(self._stable_buf)

    def get_positions(self):
        with self._lock:
            return dict(self._positions)

    def get_visibility(self):
        with self._lock:
            return dict(self._visible)

    def get_polygon(self):
        with self._lock:
            if not all(tid in self._positions for tid in CORNER_ORDER):
                return None
            return np.array([self._positions[tid] for tid in CORNER_ORDER],
                            dtype=np.int32)

    def inside_workspace(self, pt):
        poly = self.get_polygon()
        if poly is None:
            return False
        return cv2.pointPolygonTest(
            poly, (float(pt[0]), float(pt[1])), False) >= 0


# ══════════════════════════════════════════════════════════════════════════════
#  BACKGROUND YOLO THREAD
# ══════════════════════════════════════════════════════════════════════════════
class YoloWorker(threading.Thread):
    def __init__(self, model, tracker):
        super().__init__(daemon=True)
        self.model   = model
        self.tracker = tracker
        self._li     = threading.Lock()
        self._lo     = threading.Lock()
        self._frame  = None
        self._out    = []
        self._fps    = 0.0
        self._alive  = True

    def push(self, frame):
        with self._li:
            self._frame = frame.copy()

    def results(self):
        with self._lo:
            return list(self._out), self._fps

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

            t0  = time.time()
            out = []
            try:
                res = self.model(frame, conf=YOLO_CONF, verbose=False)[0]
                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cx   = (x1 + x2) // 2
                    cy   = (y1 + y2) // 2
                    if not self.tracker.inside_workspace((cx, cy)):
                        continue
                    roi        = frame[y1:y2, x1:x2]
                    color, ccf = classify_color(roi)
                    out.append({"box": (x1, y1, x2, y2), "conf": conf,
                                "color": color, "clr_conf": ccf,
                                "cx": cx, "cy": cy})
            except Exception as e:
                print(f"[YOLO] {e}")

            elapsed = time.time() - t0
            with self._lo:
                self._out = out
                self._fps = round(1.0 / elapsed, 1) if elapsed > 0 else 0
            time.sleep(max(0.0, INF_INTERVAL - elapsed))


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING
# ══════════════════════════════════════════════════════════════════════════════
CORNER_LINE_COLOR = {
    16: (0,   255, 255),   # TL — yellow
    17: (255, 200,   0),   # TR — cyan-blue
    18: (  0, 165, 255),   # BR — orange
    19: (255,   0, 255),   # BL — magenta
}
CORNER_LABEL = {16: "TL", 17: "TR", 18: "BR", 19: "BL"}


def draw_dashed_line(frame, p1, p2, color, thickness=1, dash=12):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    dist = np.hypot(x2 - x1, y2 - y1)
    n    = max(int(dist / dash), 1)
    for i in range(n):
        if i % 2 == 0:
            sx = int(x1 + (x2 - x1) * i / n)
            sy = int(y1 + (y2 - y1) * i / n)
            ex = int(x1 + (x2 - x1) * (i + 1) / n)
            ey = int(y1 + (y2 - y1) * (i + 1) / n)
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
    return dist


def draw_workspace(frame, tracker):
    poly = tracker.get_polygon()
    if poly is None:
        return
    ov = frame.copy()
    cv2.fillPoly(ov, [poly], (0, 255, 255))
    cv2.addWeighted(ov, 0.08, frame, 0.92, 0, frame)
    cv2.polylines(frame, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)

    positions = tracker.get_positions()
    visible   = tracker.get_visibility()
    for tid in CORNER_ORDER:
        if tid not in positions:
            continue
        px, py = positions[tid]
        col    = (0, 255, 0) if visible.get(tid) else (0, 170, 255)
        cv2.circle(frame, (px, py), 9, col, -1)
        cv2.putText(frame, f"ID{tid} {CORNER_LABEL[tid]}",
                    (px + 12, py - 8), FONT, 0.50, col, 2, cv2.LINE_AA)


def draw_objects(frame, detections, tracker):
    positions = tracker.get_positions()

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cx, cy = d["cx"], d["cy"]
        col    = COLOR_BGR.get(d["color"], (255, 128, 0))

        # bounding box + center dot
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.circle(frame, (cx, cy), 6, col, -1)

        # dashed lines from each corner tag center to object center
        for tid in CORNER_ORDER:
            if tid not in positions:
                continue
            tx, ty = positions[tid]
            lcol   = CORNER_LINE_COLOR[tid]
            dist   = draw_dashed_line(frame, (tx, ty), (cx, cy), lcol, 1)
            mx = (tx + cx) // 2
            my = (ty + cy) // 2
            lbl = f"{dist:.0f}px"
            cv2.putText(frame, lbl, (mx + 4, my - 4), FONT, 0.40,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, lbl, (mx + 4, my - 4), FONT, 0.40,
                        lcol, 1, cv2.LINE_AA)

        # info label box
        lines = [
            f"{d['color']}  ({d['clr_conf']:.0%})",
            f"conf: {d['conf']:.2f}",
            f"center: ({cx}, {cy})",
        ]
        lh, pad = 20, 4
        tw  = max(cv2.getTextSize(ln, FONT, 0.48, 1)[0][0] for ln in lines)
        bx2 = x1 + tw + pad * 2
        by1 = y1 - len(lines) * lh - pad * 2
        by2 = y1
        if by1 < 0:
            by1, by2 = y2, y2 + len(lines) * lh + pad * 2
        cv2.rectangle(frame, (x1, by1), (bx2, by2), col, -1)
        for i, txt in enumerate(lines):
            ty2 = by1 + pad + (i + 1) * lh - 3
            cv2.putText(frame, txt, (x1 + pad, ty2), FONT, 0.48,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, txt, (x1 + pad, ty2), FONT, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, tracker, inf_fps, cam_fps, n_obj):
    h, w   = frame.shape[:2]
    locked = tracker.is_locked()
    stab   = tracker.stable_count()
    vis    = tracker.get_visibility()
    n_vis  = sum(vis.values())

    if not locked:
        pct   = stab / STABLE_NEEDED
        bar_w = int((w - 40) * pct)
        cv2.rectangle(frame, (20, h - 38), (w - 20, h - 12), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, h - 38), (20 + bar_w, h - 12), (0, 200, 80), -1)
        missing = [f"ID{t}" for t in CORNER_ORDER if not vis.get(t)]
        msg = (f"Calibrating {stab}/{STABLE_NEEDED} frames..."
               if n_vis == 4 else f"Waiting for: {', '.join(missing)}")
        cv2.putText(frame, msg, (30, h - 17), FONT, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
    else:
        mode = "LIVE" if n_vis == 4 else "PARTIAL"
        bar  = (f"Objects:{n_obj}   Tags:{n_vis}/4 [{mode}]"
                f"   det:{inf_fps:.0f}fps  cam:{cam_fps:.0f}fps"
                f"   Q=quit  R=reset")
        cv2.rectangle(frame, (0, h - 38), (w, h), (0, 60, 0), -1)
        cv2.putText(frame, bar, (12, h - 12), FONT, 0.52,
                    (255, 255, 255), 2, cv2.LINE_AA)

    for i, tid in enumerate(CORNER_ORDER):
        col = (0, 255, 0) if vis.get(tid) else (0, 0, 200)
        cv2.circle(frame, (w - 25 - i * 22, 22), 8, col,
                   -1 if vis.get(tid) else 2)


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════
class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipe = rs.pipeline()
        cfg       = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipe.start(cfg)
        for _ in range(10):
            self.pipe.wait_for_frames()

    def read(self):
        frames = self.pipe.wait_for_frames()
        cf     = frames.get_color_frame()
        if not cf:
            return False, None
        return True, np.asanyarray(cf.get_data())

    def release(self):
        self.pipe.stop()


class OpenCVCamera:
    def __init__(self, idx=FALLBACK_CAM, width=1280, height=720):
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if REALSENSE_AVAILABLE:
        print("Connecting to RealSense D435 ...")
        try:
            cam = RealSenseCamera(1280, 720, 30)
            print("RealSense D435 ready.")
        except Exception as e:
            print(f"RealSense failed ({e}) -- using OpenCV camera.")
            cam = OpenCVCamera()
    else:
        cam = OpenCVCamera()

    print(f"Loading model: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    print("Model ready.\n")

    tracker = TagTracker()
    worker  = YoloWorker(model, tracker)
    worker.start()

    WIN = "Workspace Detector  |  Q=quit  R=reset"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    detections = []
    inf_fps    = 0.0
    cam_fps    = 0.0
    t_cam      = time.time()
    fcnt       = 0

    print("Running.  Place IDs 16(TL) 17(TR) 18(BR) 19(BL) in camera view.")
    print("Keys: Q=quit  R=reset calibration\n")

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        fcnt += 1
        now   = time.time()
        if now - t_cam >= 1.0:
            cam_fps = fcnt / (now - t_cam)
            fcnt, t_cam = 0, now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracker.update(gray)

        if tracker.is_locked():
            worker.push(frame)
            detections, inf_fps = worker.results()
        else:
            detections = []

        draw_workspace(frame, tracker)
        if tracker.is_locked():
            draw_objects(frame, detections, tracker)
        draw_hud(frame, tracker, inf_fps, cam_fps, len(detections))

        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            tracker.reset()
            detections = []
            print("Calibration reset -- re-scanning.")

    worker.stop()
    worker.join(timeout=2)
    cam.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()