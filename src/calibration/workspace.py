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
    24: "Wet Bin",       
    20: "Dangerous Bin", 
    25: "Dry Bin",       
    27: "Recycle Bin"
}

CALIB_FILE = "workspace_calibration.yaml"

ROBOT_READINGS = {
    0: (53.86880500, -355.66052200),
    1: (47.09763700, -173.95874000),
    2: (-283.01510600, -193.80304000),
    3: (-245.37525900, -371.09149200)
}

STABLE_NEEDED = 20  

CORNER_LINE_COLOR = {
    16: (0,   255, 255),   
    17: (255, 200,   0),   
    18: (  0, 165, 255),   
    19: (255,   0, 255),   
}
CORNER_LABEL = {16: "TL", 17: "TR", 18: "BR", 19: "BL"}
FONT = cv.FONT_HERSHEY_SIMPLEX

def make_aruco_detector():
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    params = aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.03
    return aruco.ArucoDetector(dictionary, params)

def detect_tags(gray_frame, detector):
    corners, ids, _ = detector.detectMarkers(gray_frame)
    found_ws = {}
    found_bins = {}
    
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            mid_int = int(mid)
            c = corners[i][0]
            
            # Calculate the exact center pixel (X, Y) of the marker
            cx = int(c[:, 0].mean())
            cy = int(c[:, 1].mean())
            
            if mid_int in TAG_ROLE:
                found_ws[mid_int] = [cx, cy]
            elif mid_int in BIN_NAMES:
                found_bins[mid_int] = [cx, cy]
                
    return found_ws, found_bins

try:
    import pyrealsense2 as prs
    prs_ready = True
except ImportError:
    prs_ready = False 

class Camera:

    def __init__(self, width=1280, height=720, fps=30):
        self.width =width
        self.height= height 
        self.fps =fps 
        self._using_prs=False 
        self._pipeline= None
        self._align= None
        self._capture= None

        self._initialize()
    
    def _initialize(self):
        if prs_ready:
            try:
                self._pipeline =prs.pipeline()
                config =prs.config()
                config.enable_stream(prs.stream.color, self.width, self.height, prs.format.bgr8, self.fps)
                config.enable_stream(prs.stream.depth, self.width, self.height, prs.format.z16, self.fps)
                self._pipeline.start(config)
                self._align=prs.align(prs.stream.color)
                self._using_prs=True
                print("[Camera] Intel RealSense initialized.")
                return
            except Exception as e:
                print(f"[Camera] RealSense init failed: {e}. Proceeding with webcame fallback.")
                self._pipeline=None
        self._capture =cv.VideoCapture(0)
        self._capture.set(cv.CAP_PROP_FRAME_WIDTH,self.width)
        self._capture.set(cv.CAP_PROP_FRAME_HEIGHT,self.height)
        self._using_prs=False
        print("[Camera] Webcam initialized.")

    def read(self):
        if self._using_prs:
            try:
                frames= self._pipeline.wait_for_frames()
                aligned = self._align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth :
                    return False, None, None
                return True, np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())
            except Exception:
                return False, None, None
            
        else :
            ok, fr = self._capture.read()
            return ok, fr, None
        
    def release(self):
        if self._using_prs and self._pipeline:
            self._pipeline.stop()
            print("[Camera] RealSense pipeline stopped.")
        elif self._capture:
            self._capture.release()
            print("[Camera] Webcam released.")
    def has_depth(self):
        return self._using_prs
    
class CoordinateMapper:

    def __init__(self, robot_readings=ROBOT_READINGS, corner_order=CORNER_ORDER, tag_to_index=TAG_TO_INDEX):
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

class BlockAnalyzer:
    blacklower =np.array([0,0,0])
    blackupper= np.array([180,255,60])
    minratio=0.02
    maxratio=0.4
    
    def has_mark(self,img):
        if img is None or img.size==0:
            return False
        hsv= cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask= cv.inRange(hsv,BlockAnalyzer.blacklower,BlockAnalyzer.blackupper)
        total=img.shape[0]*img.shape[1]
        if total==0:
            return False
        ratio=cv.countNonZero(mask)/total
        return BlockAnalyzer.minratio<ratio<BlockAnalyzer.maxratio

COLORS={
    "yellow":(0,255,255),
    "green":(0,255,0),
    "blue":(255,100,0),
    "red":(0,0,255),
    "white":(255,255,255),
    "cyan": (0,255,255),
    "magenta":(200,0,200),
    "orange":(0,165,255),
    "marker":(255,0,0),      
}
BIN_LABELS ={
    24: "Wet Bin",
    20: "Dangerous Bin",
    25: "Dry Bin",
    27: "Recycle Bin"
}
FONT= cv.FONT_HERSHEY_SIMPLEX

class HUD:
    def __init__(self,window_name,width=1400, height=800):
        self.window_name = window_name
        cv.namedWindow(self.window_name,cv.WINDOW_NORMAL)
        cv.resizeWindow(self.window_name,width,height)

    def draw_workspace(self, frame, polygon, is_live):
        if polygon is None:
            return 
        pts= np.array(polygon, np.int32).reshape((-1,1,2))
        if is_live:
            color =COLORS["green"]
            label="LIVE Workspace"
        else:
            color=COLORS["cyan"]
            label="FALLBACK Workspace"
        cv.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cv.putText(frame, f"{label}('s' to save, 'r' to reset)",(10, 30),FONT, 0.6,color,2)

    def draw_detections(self,frame,objects,highlight_color=None):    
        for obj in objects:
            x1,y1,x2,y2=obj['box']
            if obj["marked"]:
                box_color=COLORS["orange"]
            else:
                box_color=COLORS.get(obj["base_color"],COLORS["white"])
            thickness = 4 if (highlight_color and obj["color"] == highlight_color) else 2
            cv.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            cv.putText(frame, obj["color"], (x1, y1 - 10), FONT, 0.5, box_color, 2)

    def draw_bins(self, frame, bins):
        for bid, (bx, by) in bins.items():
            cv.circle(frame,(bx,by),8,COLORS["marker"], -1)
            label = BIN_LABELS.get(bid,str(bid))
            cv.putText(
                frame, f"Bin{label}",
                (bx+15,by+15),FONT,0.6,COLORS["marker"], 2
            )        

    def draw_menu(self, frame, app_state, detected_colors=None,detected_bins=None, selected_color=None,vlm_processing=False, vlm_latest=""):
        y = 60
        if app_state=="MENU":
            self._text(frame,"[1] Auto-Sort All", (20, y), COLORS["white"])
            self._text(frame,"[2] Custom Pick & Place", (20,y +30), COLORS["white"])
            self._text(frame,"[3] VLM Spatial Chat", (20,y+60), COLORS["white"])
            self._text(frame,"[q] Quit", (20,y+90), COLORS["red"])
        elif app_state == "AUTO_SORT":
            self._text(frame, "AUTO-SORTING... Press 'c' to Cancel.",
                       (20, y), COLORS["cyan"], scale=0.8)

        elif app_state == "CUSTOM_PICK_COLOR":
            if not detected_colors:
                self._text(frame, "Waiting for Cubes... | 'c' Cancel",
                           (20, y), COLORS["orange"])
            else:
                opts = " ".join([f"[{i}] {c}" for i, c in enumerate(detected_colors)])
                self._text(frame, f"Select Cube: {opts} | 'c' Cancel",
                           (20, y), COLORS["green"], scale=0.6)

        elif app_state == "CUSTOM_PICK_BIN":
            if detected_bins:
                opts = " or ".join([
                    f"[{i+1}] for {BIN_LABELS.get(b, b)}"
                    for i, b in enumerate(detected_bins)
                ])
                self._text(frame, f"Target: {selected_color}. Press {opts} | 'c' Cancel",
                           (20, y), COLORS["cyan"], scale=0.6)

        elif app_state == "CHAT":
            self._text(frame, "VLM VISION MODE ACTIVE",
                       (20, y), COLORS["magenta"], scale=0.8)
            self._text(frame, "[1] Describe Layout", (20, y + 30), COLORS["white"])
            self._text(frame, "[2] Analyze Cube Colors & Positions",
                       (20, y + 60), COLORS["white"])

            status = "ANALYZING..." if vlm_processing else "READY"
            status_clr = COLORS["red"] if vlm_processing else COLORS["green"]
            self._text(frame, f"STATUS: {status}", (20, y + 90), status_clr)
            self._text(frame, "[c] Return to Menu", (20, y + 120), COLORS["red"])

            if vlm_latest:
                truncated = vlm_latest[:100] + "..." if len(vlm_latest) > 100 else vlm_latest
                self._text(frame, f"Latest: {truncated}",
                           (20, frame.shape[0] - 30), COLORS["white"], scale=0.5)

    def draw_waiting(self, frame):
        self._text(frame, "Waiting for 4 Workspace Tags...",
                   (30, 40), COLORS["red"], scale=0.8)
    
    def show(self, frame):
        cv.imshow(self.window_name, frame)

    def _text(self,frame,text, pos,color,scale=0.7,thickness=2):
        cv.putText(frame,text, pos,FONT,scale,color,thickness)
    
    def close_all(self):
        cv.destroyAllWindows()
    

class DepthVisualizer:
    def __init__(self, scale=0.5,num_layers=10,min_depth_mm=100,max_depth_mm=2500):
        self.scale=scale
        self.num_layers=num_layers
        self.min_depth=min_depth_mm
        self.max_depth=max_depth_mm
        self._layer_step=256//num_layers

    
    def render(self, color_img,depth_img,objects=None):
        
        if objects is None:
            objects =[]

        h =int(depth_img.shape[0]* self.scale)
        w = int(depth_img.shape[1]*self.scale)


        color_small= cv.resize(color_img,(w,h))
        depth_small =cv.resize(depth_img,(w,h))

        clipped = np.clip(depth_small, self.min_depth, self.max_depth)
        normalized= cv.normalize(clipped, None,0,255,cv.NORM_MINMAX, dtype=cv.CV_8U)

        normalized = 255 - normalized
        normalized[depth_small ==0] =0  
        layered = (normalized//self._layer_step) *self._layer_step

        colormap =cv.applyColorMap(layered,cv.COLORMAP_TURBO)
        colormap[depth_small ==0] = [0,0,0]  

        edges = cv.Canny(layered,50,150)
        colormap[edges>0] = [0,255,0]

        for obj in objects:
            sx1,sy1, sx2,sy2 =[int(v *self.scale) for v in obj["box"]]
            scx=int(obj["cx"]*self.scale)
            scy=int(obj["cy"]*self.scale)

            depth_mm = depth_img[obj["cy"], obj["cx"]]

            cv.rectangle(colormap, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)
            cv.circle(colormap, (scx, scy), 3, (0, 0, 255), -1)
            cv.putText(
                colormap, f"{obj['color']} ({depth_mm}mm)",
                (sx1, sy1 - 5), FONT, 0.4, (255, 255, 255), 1
            )

        cv.putText(
            colormap, "TOPOGRAPHIC DEPTH LAYERS",
            (10, 25), FONT, 0.6, COLORS["cyan"], 2
        )

        return np.hstack((color_small, colormap))
