import cv2 as cv
import numpy as np

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
        # webcam fallback system
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
