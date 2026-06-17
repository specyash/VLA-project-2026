import cv2 as cv
import numpy as np
from src.camera.camera import Camera
from src.perception.yolo_detector import YOLODetector
from src.perception.block_analyzer import BlockAnalyzer
from src.calibration.workspace import Workspace, CORNER_ORDER, TAG_TO_INDEX, ROBOT_READINGS, CALIB_FILE, make_aruco_detector, detect_tags
from src.calibration.coordinate_mapper import CoordinateMapper
from src.robot.xarm_robot import XArmRobotAdapter
from src.app.hud import HUD
from src.app.depth_visualizer import DepthVisualizer

def main():
    print("[System] Initializing integration pipeline...")
    
    # 1. Initialize Camera Source
    camera = Camera()
    
    # 2. Initialize YOLO Detector & HSV Block Analyzer
    detector = YOLODetector()
    analyzer = BlockAnalyzer()
    
    # 3. Initialize Workspace Calibration & Coordinate Mapper
    workspace = Workspace(calib_file=CALIB_FILE, corner_order=CORNER_ORDER)
    aruco_detector = make_aruco_detector()
    mapper = CoordinateMapper(
        robot_readings=ROBOT_READINGS, 
        corner_order=CORNER_ORDER, 
        tag_to_index=TAG_TO_INDEX
    )
    
    # 4. Initialize Robot (dry_run=True for safety during development/simulation)
    robot = XArmRobotAdapter(dry_run=True)
    
    # 5. Initialize HUD and Depth Visualizer
    hud = HUD(window_name="Main View")
    depth_vis = DepthVisualizer()
    cv.namedWindow("TECHY SENSOR HUD", cv.WINDOW_NORMAL)
    cv.resizeWindow("TECHY SENSOR HUD", 1400, 450)

    print("[System] Initialization complete.")

    try:
        while True:
            # A. Read frame from camera
            ok, frame, depth_frame = camera.read()
            if not ok or frame is None:
                continue

            # B. Convert to grayscale and run ArUco detection
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            found_ws, found_bins = detect_tags(gray, aruco_detector)
            workspace.update(found_ws, found_bins)

            # C. Update coordinate mapping (homography) with workspace corners
            active_ws_pixels = workspace.get_workspace_pixels()
            mapper.update(active_ws_pixels)

            # D. Object Detection and HSV Black Marker Classification
            valid_objects = []
            hud_objects = []
            raw_detections = detector.detect(frame)
            for obj in raw_detections:
                cx, cy = obj.center_px
                # If workspace is ready, filter out objects outside boundary.
                # If not ready, keep them so we can test the camera and HUD.
                if not workspace.is_ready() or workspace.contains(cx, cy):
                    # Crop ROI for HSV black marker checking
                    x, y, w, h = obj.box
                    y1, y2 = max(0, y), min(frame.shape[0], y + h)
                    x1, x2 = max(0, x), min(frame.shape[1], x + w)
                    roi = frame[y1:y2, x1:x2]
                    
                    obj.marked = analyzer.has_mark(roi)
                    obj.label = f"{obj.base_color}_marked" if obj.marked else f"{obj.base_color}_plain"
                    
                    valid_objects.append(obj)
                    
                    # Prepare dictionary for HUD drawing
                    hud_objects.append({
                        "box": (x, y, x + w, y + h),
                        "marked": obj.marked,
                        "base_color": obj.base_color,
                        "color": obj.label,
                        "cx": cx,
                        "cy": cy
                    })

            # E. Show frame with HUD overlays (workspace boundary, bins & detections)
            annotated_frame = frame.copy()
            if workspace.is_ready():
                hud.draw_workspace(annotated_frame, workspace.get_polygon(), workspace.is_live())
                hud.draw_bins(annotated_frame, workspace.get_bins())
            else:
                hud.draw_waiting(annotated_frame)
            
            # Always draw detections on the frame for visual feedback
            hud.draw_detections(annotated_frame, hud_objects)

            hud.show(annotated_frame)

            # F. Render and show Topographic Depth layers
            if depth_frame is not None:
                depth_hud = depth_vis.render(frame, depth_frame, hud_objects)
                cv.imshow("TECHY SENSOR HUD", depth_hud)

            # E. Handle keyboard inputs
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[System] Shutting down...")
                break
            elif key == ord('s') and workspace.is_live():
                # Save current live tag positions as fallback calibration
                workspace.save()
            elif key == ord('r'):
                # Reset saved tag calibration
                workspace.reset_calibration()

    except Exception as e:
        print(f"[System Error] Exception in main loop: {e}")
    finally:
        # D. Clean up all resources
        print("[System] Releasing camera resources...")
        camera.release()
        hud.close_all()
        robot.disconnect()
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()

