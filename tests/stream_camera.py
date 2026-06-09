import cv2 as cv
import numpy as np
import sys
import os

# Ensure the root directory is in the import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera.camera import Camera


def main():
    print("[Camera Test] Initializing Camera class...")
    cam = Camera(width=1280, height=720, fps=30)
    
    print(f"[Camera Test] Camera initialized successfully.")
    print(f"[Camera Test] RealSense Depth Available: {cam.has_depth()}")
    print("\nControls:")
    print("  - Press 'q' to quit the stream")
    print("  - Press 's' to save a frame snapshot (color & depth) to disk\n")
    
    window_name = "Camera Stream Test"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    
    if cam.has_depth():
        cv.resizeWindow(window_name, 1920, 720) # Side-by-side wide window
    else:
        cv.resizeWindow(window_name, 1280, 720)

    try:
        while True:
            ok, color, depth = cam.read()
            if not ok or color is None:
                print("[Camera Test] Error: Failed to read frame.")
                break
                
            # If depth is available, visualize it next to the color frame
            if depth is not None:
                # Clip depth to reasonable range (200mm to 2000mm) for coloring contrast
                clipped = np.clip(depth, 200, 2000)
                normalized = cv.normalize(clipped, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
                depth_colored = cv.applyColorMap(normalized, cv.COLORMAP_JET)
                
                # Combine color frame and colored depth frame side-by-side
                display_frame = np.hstack((color, depth_colored))
            else:
                display_frame = color
                
            # Render visual HUD overlays
            mode_text = "RealSense (Color + Depth)" if cam.has_depth() else "Webcam (Fallback Mode)"
            cv.putText(display_frame, f"Mode: {mode_text}", (30, 50), 
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(display_frame, "Press 'q' to quit, 's' to capture", (30, display_frame.shape[0] - 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
            
            cv.imshow(window_name, display_frame)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv.imwrite("snapshot_color.jpg", color)
                print("[Camera Test] Captured snapshot_color.jpg")
                if depth is not None:
                    np.save("snapshot_depth.npy", depth)
                    cv.imwrite("snapshot_depth_viz.jpg", depth_colored)
                    print("[Camera Test] Captured snapshot_depth.npy & snapshot_depth_viz.jpg")
                
    finally:
        cam.release()
        cv.destroyAllWindows()
        print("[Camera Test] Camera resources released and windows closed.")


if __name__ == "__main__":
    main()
