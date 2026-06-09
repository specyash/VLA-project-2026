import cv2 as cv
import numpy as np

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

        # scaling down the imgs...
        color_small= cv.resize(color_img,(w,h))
        depth_small =cv.resize(depth_img,(w,h))
        
        # normalise the depth to 255
        clipped = np.clip(depth_small, self.min_depth, self.max_depth)
        normalized= cv.normalize(clipped, None,0,255,cv.NORM_MINMAX, dtype=cv.CV_8U)

        normalized = 255 - normalized
        normalized[depth_small ==0] =0  # Invalid depth stays black

        #convert to layers
        layered = (normalized//self._layer_step) *self._layer_step

        #applying turbo colormap for the heatmap look
        colormap =cv.applyColorMap(layered,cv.COLORMAP_TURBO)
        colormap[depth_small ==0] = [0,0,0]  # black out invalid pixels... 

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

        # to setup title 
        cv.putText(
            colormap, "TOPOGRAPHIC DEPTH LAYERS",
            (10, 25), FONT, 0.6, COLORS["cyan"], 2
        )

        # stacking side by side...
        return np.hstack((color_small, colormap))
