import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import threading
import time
import yaml
import os

ARUCO_DICT = aruco.DICT_APRILTAG_36h11

TAG_ROLE = {
    16: "top_left",
    17: "top_right",
    18: "bottom_right",
    19: "bottom_left",
}
CORNER_ORDER = [16, 17, 18, 19] 
TAG_TO_INDEX = {16: 0, 17: 1, 18: 2, 19: 3}

BIN_NAMES = {
    24: "Wet Bin",       # Green
    20: "Dangerous Bin", # Red
    25: "Dry Bin",       # Blue
    27: "Recycle Bin"    # Yellow
}

CALIB_FILE = "workspace_calibration.yaml"

ROBOT_READINGS = {
    0: (53.86880500, -355.66052200),
    1: (47.09763700, -173.95874000),
    2: (-283.01510600, -193.80304000),
    3: (-245.37525900, -371.09149200)
}

def make_aruco_detector():
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.03
    return aruco.ArucoDetector(dictionary, params)

class CoordinateMapper:

    def __init__(self, robot_readings,corner_order, tag_to_index):
        self._robot_readings= robot_readings
        self._corner_order =corner_order
        self._tag_to_index= tag_to_index
        self._homography =None

    def update(self, workspace_pixels):
        if not workspace_pixels or len(workspace_pixels)!=4 :
            self._homography=None
            return 
        srcpts = []
        dstpts = []
        for tag_id in self._corner_order:
            px, py =workspace_pixels[tag_id]
            srcpts.append([px,py])
            idx = self._tag_to_index[tag_id]
            rx, ry = self._robot_readings[idx]
            dstpts.append([rx,ry])
        src = np.array(srcpts, dtype=np.float32)
        dst = np.array(dstpts, dtype=np.float32)
        self._homography, _ = cv.findHomography(src, dst)
    
    def cvt2robot(self, px,py):
        if self._homography is None:
            return None
        point = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv.perspectiveTransform(point, self._homography)
        robot_x, robot_y = transformed[0][0]
        return float(robot_x), float(robot_y)
    
    def is_valid(self):
        return self._homography is not None

class Workspace:
    
    def __init__(self, calib_file, corner_order):
        self._calib_file =calib_file
        self._corner_order= corner_order
        self._live_workspace=None
        self._live_bins= {}
        self._saved_workspace={}
        self._saved_bins={}
        self._load_calibration()

    def _load_calibration(self):
        if not os.path.exists(self._calib_file):
            return
        try:
            with open(self._calib_file,'r') as f:
                data= yaml.safe_load(f)
            if data and 'workspace_pixels'  in data:
                self._saved_workspace={int(k): tuple(v) for k, v in data['workspace_pixels'].items()}
                print(f"[Workspace] Loaded calibration from {self._calib_file}")

            if data and "bin_pixels" in data:
                self._saved_bins={int(k): tuple(v) for k, v in data['bin_pixels'].items()}
            print(f"[Workspace] Loaded calibration from {self._calib_file}")
        except Exception as e:
            print(f"[Workspace] Failed to load calibration: {e}")

    def save(self):
        data={}
        ws= self._live_workspace or self._saved_workspace
        bins= self._live_bins or self._saved_bins
        if ws:
            data['workspace_pixels']={k: list(v) for k, v in ws.items()}
        if bins:
            data['bin_pixels']={k: list(v) for k, v in bins.items()}
        try:
            with open(self._calib_file, 'w') as f:
                yaml.dump(data, f)
            if ws:
                self._saved_workspace = ws.copy()
            if bins:
                self._saved_bins=bins.copy()
        except Exception as e:
            print(f"[Workspace] Save failed: {e}")
            return False
    
    def reset_calibration(self):
        self._saved_workspace={}
        self._saved_bins={}
        if os.path.exists(self._calib_file):
            os.remove(self._calib_file)
            print(f"[Workspace] Calibration file deleted.")

    def update(self,workspace_tags, bin_tags):
        self._live_workspace =workspace_tags if (workspace_tags and len(workspace_tags) == 4) else None
        self._live_bins=bin_tags or {}
    
    def get_workspace_pixels(self):
        return self._live_workspace or self._saved_workspace
    
    def get_bins(self):
        merged= self._saved_bins.copy()
        for bid,pos in self._live_bins.items():
            merged[bid]=pos
        return merged

    def get_polygon(self):
        ws= self.get_workspace_pixels()
        if not ws or len(ws)!=4:
            return None
        points = [ws[tid] for tid in self._corner_order]
        return np.array(points, dtype=np.float32)
    
    def is_live(self):
        return self._live_workspace is not None

    def is_ready(self):
        return self.get_polygon() is not None

    def contains(self,px,py,margin= 75):
        polygon= self.get_polygon()
        if polygon is None:
            return False 
        moments= cv.moments(polygon)
        if moments['m00']== 0: 
            return False 
        cx= moments['m10']/moments['m00']
        cy= moments['m01']/moments['m00']
        expanded= []
        for pt in polygon:
            dx= pt[0]-cx
            dy=pt[1]-cy
            distance=max(np.hypot(dx, dy),1.0)
            scale_factor=(distance + margin)/distance
            expanded.append((cx+dx*scale_factor,cy + dy*scale_factor))
        expanded_polygon = np.array(expanded, dtype=np.float32)
        return cv.pointPolygonTest(expanded_polygon, (float(px), float(py)), False) >= 0
