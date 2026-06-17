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
from src.planning.pick_place import create_pick_place_plan, execute_pick_place_plan
from src.config import DROP_Z

COLOR_TO_BIN_MAP = {
    "green": 24, "red": 20, "blue": 25, "yellow": 27
}

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

    # State machine variables
    app_state = "MENU"
    selected_color = None
    empty_frames = 0
    bin_offsets = {}

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

            # D. Object Detection and HSV Black Marker Classification (Runs always for testing)
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
            detected_colors = sorted(list(set(obj.label for obj in valid_objects)))
            active_bins = workspace.get_bins()
            detected_bins = sorted(list(active_bins.keys()))

            if workspace.is_ready():
                hud.draw_workspace(annotated_frame, workspace.get_polygon(), workspace.is_live())
                hud.draw_bins(annotated_frame, active_bins)
                # Draw interactive menu overlay
                hud.draw_menu(annotated_frame, app_state, detected_colors, detected_bins, selected_color)
            else:
                hud.draw_waiting(annotated_frame)
            
            # Always draw detections on the frame for visual feedback
            hud.draw_detections(annotated_frame, hud_objects)
            hud.show(annotated_frame)

            # F. Render and show Topographic Depth layers
            if depth_frame is not None:
                depth_hud = depth_vis.render(frame, depth_frame, hud_objects)
                cv.imshow("TECHY SENSOR HUD", depth_hud)

            # G. Handle keyboard inputs & State Machine Transitions
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
                bin_offsets.clear()
            
            # Run state machine logic only if workspace is calibrated and ready
            if workspace.is_ready():
                if app_state == "MENU":
                    if key == ord('1'):
                        app_state = "AUTO_SORT"
                        print("[System] Auto-Sort state engaged.")
                    elif key == ord('2'):
                        app_state = "CUSTOM_PICK_COLOR"
                        print("[System] Custom Pick-Place engaged. Select a cube color index.")
                
                elif app_state == "AUTO_SORT":
                    if key == ord('c'):
                        app_state = "MENU"
                        empty_frames = 0
                        print("[System] Auto-Sort cancelled.")
                    else:
                        # Find a target cube that can be sorted
                        target = None
                        for obj in valid_objects:
                            bin_id = COLOR_TO_BIN_MAP.get(obj.base_color)
                            if bin_id in active_bins:
                                target = obj
                                break
                        
                        if target:
                            empty_frames = 0
                            bin_id = COLOR_TO_BIN_MAP[target.base_color]
                            
                            # 1. Transform coordinates to robot base space
                            obj_rx, obj_ry = mapper.cvt2robot(target.center_px[0], target.center_px[1])
                            bin_px, bin_py = active_bins[bin_id]
                            bin_rx, bin_ry = mapper.cvt2robot(bin_px, bin_py)
                            
                            if obj_rx is not None and bin_rx is not None:
                                # Get actual object Z height from depth sensor if available
                                obj_rz = float(depth_frame[target.center_px[1], target.center_px[0]]) if depth_frame is not None else 176.0
                                if obj_rz <= 0.0 or obj_rz > 2000.0:  # Check for invalid depth readings
                                    obj_rz = 176.0
                                
                                # Stacking offset tracking
                                current_offset = bin_offsets.get(bin_id, 0.0)
                                drop_z = 176.0 + current_offset
                                
                                # 2. Generate Plan
                                plan = create_pick_place_plan(
                                    object_xyz=(obj_rx, obj_ry, obj_rz),
                                    bin_xy=(bin_rx, bin_ry),
                                    drop_z=drop_z
                                )
                                
                                # 3. Execute
                                print(f"[System] Sorting {target.label} (Z={obj_rz:.1f}mm) into Bin {bin_id} (Drop Z={drop_z:.1f}mm)...")
                                execute_pick_place_plan(robot, plan)
                                
                                # Update stacking memory offset
                                bin_offsets[bin_id] = current_offset + 25.0
                                
                                # Clear video buffer after execution
                                for _ in range(10):
                                    camera.read()
                        else:
                            empty_frames += 1
                            if empty_frames > 20:
                                print("[System] Workspace is clear. Returning to Menu.")
                                app_state = "MENU"
                                empty_frames = 0
                                
                elif app_state == "CUSTOM_PICK_COLOR":
                    if key == ord('c'):
                        app_state = "MENU"
                        print("[System] Custom Pick cancelled.")
                    else:
                        # Check if user selected one of the visible block colors
                        for idx, color_name in enumerate(detected_colors):
                            if key == ord(str(idx)):
                                selected_color = color_name
                                app_state = "CUSTOM_PICK_BIN"
                                print(f"[System] Selected cube: {selected_color}. Select destination bin index.")
                                break
                                
                elif app_state == "CUSTOM_PICK_BIN":
                    if key == ord('c'):
                        app_state = "MENU"
                        selected_color = None
                        print("[System] Custom Pick cancelled.")
                    else:
                        # Check if user selected one of the active bins
                        for idx, bin_id in enumerate(detected_bins):
                            # User input is 1-indexed to avoid conflict with color selection index (which is 0-indexed)
                            if key == ord(str(idx + 1)):
                                # Find target object matching the selected color
                                target = next((o for o in valid_objects if o.label == selected_color), None)
                                if target:
                                    # 1. Coordinate transformations
                                    obj_rx, obj_ry = mapper.cvt2robot(target.center_px[0], target.center_px[1])
                                    bin_px, bin_py = active_bins[bin_id]
                                    bin_rx, bin_ry = mapper.cvt2robot(bin_px, bin_py)
                                    
                                    if obj_rx is not None and bin_rx is not None:
                                        obj_rz = float(depth_frame[target.center_px[1], target.center_px[0]]) if depth_frame is not None else 176.0
                                        if obj_rz <= 0.0 or obj_rz > 2000.0:
                                            obj_rz = 176.0
                                            
                                        current_offset = bin_offsets.get(bin_id, 0.0)
                                        drop_z = 176.0 + current_offset
                                        
                                        # 2. Generate and Execute Plan
                                        plan = create_pick_place_plan(
                                            object_xyz=(obj_rx, obj_ry, obj_rz),
                                            bin_xy=(bin_rx, bin_ry),
                                            drop_z=drop_z
                                        )
                                        print(f"[System] Custom Pick: Sorting {target.label} to Bin {bin_id}...")
                                        execute_pick_place_plan(robot, plan)
                                        
                                        bin_offsets[bin_id] = current_offset + 25.0
                                        
                                        for _ in range(10):
                                            camera.read()
                                            
                                app_state = "MENU"
                                selected_color = None
                                break

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

