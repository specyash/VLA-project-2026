import cv2 as cv
import numpy as np
import yaml
import os

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