import cv2 as cv
import numpy as np
from src.camera.camera import Camera

def main():
    print("[ArUco Test] Initializing camera...")
    camera = Camera()
    
    print("[ArUco Test] Setting up AprilTag 36h11 detector...")
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_36h11)
    params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, params)
    
    print("[ArUco Test] Starting stream. Press 'q' to quit.")
    
    try:
        while True:
            # 1. Read camera frame
            ok, frame, _ = camera.read()
            if not ok or frame is None:
                continue
                
            # 2. Convert to grayscale (required for ArUco decoding)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # 3. Detect markers
            corners, ids, _ = detector.detectMarkers(gray)
            
            # 4. Annotate and draw on frame
            annotated_frame = frame.copy()
            if ids is not None and len(ids) > 0:
                # OpenCV helper to draw borders and print ID text next to the markers
                cv.aruco.drawDetectedMarkers(annotated_frame, corners, ids)
                
                # Print coordinates and IDs to console
                for index, marker_id in enumerate(ids.flatten()):
                    c = corners[index][0]
                    cx = int(c[:, 0].mean())
                    cy = int(c[:, 1].mean())
                    print(f"Tag ID: {marker_id} detected at pixel: ({cx}, {cy})")
            else:
                cv.putText(
                    annotated_frame, 
                    "No tags detected. Show AprilTag 36h11 markers.", 
                    (20, 40), 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
            
            # 5. Display frame
            cv.imshow("ArUco Tag Test", annotated_frame)
            
            # 6. Exit check
            if cv.waitKey(1) & 0xFF == ord('q'):
                print("[ArUco Test] Shutting down...")
                break
                
    except Exception as e:
        print(f"[ArUco Test Error]: {e}")
    finally:
        camera.release()
        cv.destroyAllWindows()
        print("[ArUco Test] Resources released. Closed.")

if __name__ == "__main__":
    main()
