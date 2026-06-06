import pyrealsense2 as rs
import numpy as np
import cv2
import ollama
import base64
import pyttsx3
import matplotlib.pyplot as plt
import threading
import time
import re
from mpl_toolkits.mplot3d import Axes3D

# --- CONFIGURATION ---
REMOTE_OLLAMA_IP = "192.168.239.57" 
MODEL_NAME = "llama3.2-vision"
# The "God Prompt" for one-go spatial scene graphing
PROMPT = (
    "Identify all objects in this scene. For each, give its color and "
    "its precise spatial relationship to other objects (e.g., 'left of', 'on top of', 'behind'). "
    "Output the result as a natural paragraph. Do not use any markdown, asterisks, or special characters."
)

class AutonomousSpatialVLM:
    def __init__(self):
        # State Management
        self.last_capture_time = 0
        self.is_processing = False
        self.latest_insight = "Initializing..."
        
        # TTS Setup (Making it sound more natural)
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        # Try to pick a more natural sounding voice if available (index 1 is often female/softer)
        self.engine.setProperty('voice', voices[1].id if len(voices) > 1 else voices[0].id)
        self.engine.setProperty('rate', 160) # Slightly slower for clarity
        self.engine.setProperty('volume', 1.0)
        
        # RealSense Setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(self.config)
        
        # Ollama Setup
        self.client = ollama.Client(host=f'http://{REMOTE_OLLAMA_IP}:11434')
        
        # Matplotlib Setup
        plt.ion()
        self.fig = plt.figure(figsize=(7, 5))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def clean_text(self, text):
        """Removes markdown artifacts for natural speech."""
        # Remove asterisks, hashtags, underscores, and multiple spaces
        text = re.sub(r'[*#_]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def speak(self, text):
        cleaned = self.clean_text(text)
        print(f"\n[Scene Insight]: {cleaned}")
        self.engine.say(cleaned)
        self.engine.runAndWait()

    def get_spatial_points(self, depth_frame):
        """Downsampled point cloud for the graph."""
        stride = 20 
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        points = []
        for y in range(0, 480, stride):
            for x in range(0, 640, stride):
                dist = depth_frame.get_distance(x, y)
                if 0.1 < dist < 2.5:
                    pt = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], dist)
                    points.append(pt)
        return np.array(points)

    def vlm_thread_worker(self, frame, depth_frame):
        """Background worker to prevent UI freezing."""
        self.is_processing = True
        try:
            # 1. Image Encoding
            _, buffer = cv2.imencode('.jpg', frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            
            # 2. VLM Inference
            res = self.client.generate(model=MODEL_NAME, prompt=PROMPT, images=[img_str])
            self.latest_insight = res['response']
            
            # 3. Update Spatial Plot
            pts = self.get_spatial_points(depth_frame)
            self.update_3d_plot(pts)
            
            # 4. Voice Feedback
            self.speak(self.latest_insight)
            
        except Exception as e:
            print(f"Worker Error: {e}")
        finally:
            self.is_processing = False
            self.last_capture_time = time.time()

    def update_3d_plot(self, points):
        self.ax.clear()
        if len(points) > 0:
            self.ax.scatter(points[:, 0], points[:, 2], -points[:, 1], c=points[:, 2], cmap='winter', s=2)
        self.ax.set_title("Dynamic Spatial Scene Graph")
        plt.draw()
        plt.pause(0.001)

    def draw_overlay(self, frame):
        """Heads-up display showing current system status."""
        h, w, _ = frame.shape
        # Top Bar
        cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
        status_color = (0, 0, 255) if self.is_processing else (0, 255, 0)
        status_text = "VLM ANALYZING..." if self.is_processing else "SENSORS ACTIVE"
        cv2.putText(frame, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Bottom text block for the last insight
        cv2.rectangle(frame, (0, h-80), (w, h), (0, 0, 0), -1)
        wrapped_text = self.latest_insight[:150] + "..." if len(self.latest_insight) > 150 else self.latest_insight
        cv2.putText(frame, wrapped_text, (15, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        print("Autonomous Spatial Assistant is Running...")
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                img = np.asanyarray(color_frame.get_data())
                
                # Check for 2-second interval and ensure not already processing
                if time.time() - self.last_capture_time > 2.0 and not self.is_processing:
                    # Launch VLM task in a background thread
                    t = threading.Thread(target=self.vlm_thread_worker, args=(img.copy(), depth_frame))
                    t.daemon = True
                    t.start()

                # Visuals
                self.draw_overlay(img)
                cv2.imshow("Spatial Assistant", img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    assistant = AutonomousSpatialVLM()
    assistant.run()