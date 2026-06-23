import cv2 as cv
import numpy as np
from src.camera.camera import Camera
from src.perception.yolo_detector import YOLODetector
from src.perception.block_analyzer import BlockAnalyzer
from src.calibration.workspace import Workspace, CORNER_ORDER, TAG_TO_INDEX, ROBOT_READINGS, CALIB_FILE, make_aruco_detector,detect_tags
from src.calibration.coordinate_mapper import KabschCoordinateMapper
from src.robot.xarm_robot import XArmRobotAdapter
from src.app.hud import HUD
from src.app.depth_visualizer import DepthVisualizer
from src.planning.pick_place import create_pick_place_plan, execute_pick_place_plan
from src.config import DROP_Z, KABSCH_R, KABSCH_t, CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY, PICK_Z, PICK_Z_OFFSET, SURFACE_DEPTH, DROP_Z_OFFSET

COLOR_TO_BIN_MAP = {
    "green":24,
    "red": 20,
    "blue": 25,
    "yellow": 27
}

def main():
    camera = None
    robot = None
    hud = None

    print("[System] initializing integration pipeline...")
    
    # Initialize robot first
    robot = XArmRobotAdapter(dry_run=False)
    
    # Move robot to home position
    print("[System] Moving robot to home position...")
    robot.home()
    
    # Prompt user to continue
    try:
        user_input = input("\n[System] Type 'yes' to start the sorting loop: ").strip().lower()
        if user_input != 'yes':
            print("[System] Startup aborted by user. Exiting.")
            return
    except (KeyboardInterrupt, EOFError):
        print("\n[System] Startup cancelled. Exiting.")
        return

    # Initialize the rest of the workspace
    camera = Camera()
    detector = YOLODetector()
    analyzer = BlockAnalyzer()

    workspace = Workspace(calib_file=CALIB_FILE, corner_order=CORNER_ORDER)
    mapper = KabschCoordinateMapper(KABSCH_R, KABSCH_t, CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY)
    hud = HUD(window_name="Main View")
    depth_vis = DepthVisualizer()
    cv.namedWindow("TECHY SENSOR HUD", cv.WINDOW_NORMAL)
    cv.resizeWindow("TECHY SENSOR HUD", 1400, 450)
    aruco_detector = make_aruco_detector()
    
    # State machine variables
    app_state = "MENU"
    selected_color = None
    empty_frames = 0
    bin_offsets = {}

    print("[System] Initialization completed")

    try:
        while True:
            ok, frame, depth_frame = camera.read()
            if not ok or frame is None:
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            found_ws, found_bins = detect_tags(gray, aruco_detector)
            workspace.update(found_ws, found_bins)

            # (No homography update needed; Kabsch calibration is static)

            # D. Object Detection and HSV Black Marker Classification
            valid_objects = []
            hud_objects = []
            if workspace.is_ready():
                raw_detections = detector.detect(frame)
                for obj in raw_detections:
                    cx, cy = obj.center_px
                    if workspace.contains(cx, cy):
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


            annotated_frame = frame.copy()
            detected_colors = sorted(list(set(obj.label for obj in valid_objects)))
            active_bins = workspace.get_bins()
            detected_bins = sorted(list(active_bins.keys()))

            if workspace.is_ready():
                hud.draw_workspace(annotated_frame, workspace.get_polygon(), workspace.is_live())
                hud.draw_bins(annotated_frame, workspace.get_bins())
                hud.draw_menu(annotated_frame, app_state, detected_colors, detected_bins, selected_color)
            else:
                hud.draw_waiting(annotated_frame)

            hud.draw_detections(annotated_frame, hud_objects)
            hud.show(annotated_frame)

            # F. Render and show Topographic Depth layers
            if depth_frame is not None:
                depth_hud = depth_vis.render(frame, depth_frame, hud_objects)
                cv.imshow("TECHY SENSOR HUD", depth_hud)

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

            #sorting machine logic

            if workspace.is_ready():
                if app_state == "MENU":
                    if key == ord('1'):
                        app_state = "AUTO_SORT"
                        print("[System] Auto-sort sequence engaged")
                    elif key == ord('2'):
                        app_state = "CUSTOM_PICK_COLOR"
                        print("[System] Custom Pick sequence engaged, Select a cube color index")

                elif app_state == "AUTO_SORT":
                    if key == ord("c"):
                        app_state = "MENU"
                        empty_frames = 0
                        print("[System] Auto sort cancelled")
                    else:
                        target = None
                        for obj in valid_objects:
                            bin_id = COLOR_TO_BIN_MAP.get(obj.base_color)
                            if bin_id in active_bins:
                                target = obj
                                break
                        if target:
                            empty_frames = 0
                            bin_id = COLOR_TO_BIN_MAP.get(target.base_color)
                            # Get depth for object and bin
                            obj_d = float(depth_frame[target.center_px[1], target.center_px[0]]) if depth_frame is not None else 0.0
                            if obj_d <= 0.0 or obj_d > 2000.0:
                                print(f"[System Error] Depth unreadable ({obj_d:.1f}mm) for target {target.label}. Skipping pick to prevent collision.")
                                continue
                                
                            bin_px, bin_py = active_bins[bin_id]
                            bin_d = float(depth_frame[bin_py, bin_px]) if depth_frame is not None else 0.0
                            if bin_d <= 0.0 or bin_d > 2000.0:
                                bin_d = SURFACE_DEPTH

                            # Convert to 3D Robot Base Frame
                            obj_rx, obj_ry, obj_rz = mapper.cvt2robot_3d(target.center_px[0], target.center_px[1], obj_d)
                            bin_rx, bin_ry, bin_rz = mapper.cvt2robot_3d(bin_px, bin_py, bin_d)
                            
                            if obj_rz <= -100.0 or obj_rz > 500.0:
                                obj_rz = PICK_Z + PICK_Z_OFFSET
                            if bin_rz <= -100.0 or bin_rz > 500.0:
                                bin_rz = PICK_Z

                            # Calculate the actual height of this specific block
                            measured_height = obj_rz - PICK_Z
                            if 10.0 <= measured_height <= 80.0:
                                block_height = measured_height
                            else:
                                block_height = 25.0  # Fallback to default block height
                            
                            current_offset = bin_offsets.get(bin_id, 0.0)
                            drop_z = bin_rz + block_height + current_offset + DROP_Z_OFFSET
                            
                            plan = create_pick_place_plan(
                                object_xyz=(obj_rx, obj_ry, obj_rz),
                                bin_xy=(bin_rx, bin_ry),
                                drop_z=drop_z
                            )

                            print(f"[System] Sorting {target.label} (Z={obj_rz:.1f}mm, Measured Height={block_height:.1f}mm) into Bin {bin_id} (Drop Z={drop_z:.1f}mm)")
                            execute_pick_place_plan(robot, plan)
                            bin_offsets[bin_id] = current_offset + block_height

                            for _ in range(10):
                                camera.read()
                        else:
                            empty_frames += 1
                            if empty_frames > 20:
                                print("[System] Workspace is clear. Returning to menu")
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
                                    # Get depth for object and bin
                                    obj_d = float(depth_frame[target.center_px[1], target.center_px[0]]) if depth_frame is not None else 0.0
                                    if obj_d <= 0.0 or obj_d > 2000.0:
                                        print(f"[System Error] Depth unreadable ({obj_d:.1f}mm) for target {target.label}. Aborting custom pick.")
                                        app_state = "MENU"
                                        selected_color = None
                                        break
                                        
                                    bin_px, bin_py = active_bins[bin_id]
                                    bin_d = float(depth_frame[bin_py, bin_px]) if depth_frame is not None else 0.0
                                    if bin_d <= 0.0 or bin_d > 2000.0:
                                        bin_d = SURFACE_DEPTH

                                    # 1. Coordinate transformations
                                    obj_rx, obj_ry, obj_rz = mapper.cvt2robot_3d(target.center_px[0], target.center_px[1], obj_d)
                                    bin_rx, bin_ry, bin_rz = mapper.cvt2robot_3d(bin_px, bin_py, bin_d)
                                    
                                    if obj_rz <= -100.0 or obj_rz > 500.0:
                                        obj_rz = PICK_Z + PICK_Z_OFFSET
                                    if bin_rz <= -100.0 or bin_rz > 500.0:
                                        bin_rz = PICK_Z
                                    
                                    # Calculate the actual height of this specific block
                                    measured_height = obj_rz - PICK_Z
                                    if 10.0 <= measured_height <= 80.0:
                                        block_height = measured_height
                                    else:
                                        block_height = 25.0  # Fallback to default block height
                                            
                                    current_offset = bin_offsets.get(bin_id, 0.0)
                                    drop_z = bin_rz + block_height + current_offset + DROP_Z_OFFSET
                                    
                                    # 2. Generate and Execute Plan
                                    plan = create_pick_place_plan(
                                        object_xyz=(obj_rx, obj_ry, obj_rz),
                                        bin_xy=(bin_rx, bin_ry),
                                        drop_z=drop_z
                                    )
                                    print(f"[System] Custom Pick: Sorting {target.label} (Z={obj_rz:.1f}mm, Measured Height={block_height:.1f}mm) to Bin {bin_id}...")
                                    execute_pick_place_plan(robot, plan)
                                    
                                    bin_offsets[bin_id] = current_offset + block_height
                                    
                                    for _ in range(10):
                                        camera.read()
                                        
                                app_state = "MENU"
                                selected_color = None
                                break
            
    except Exception as e:
        print(f"[System Error] Exception in main loop: {e}")
    finally:
        print("[System] Releasing camera resources...")
        if camera is not None:
            camera.release()
        if hud is not None:
            hud.close_all()
        if robot is not None:
            robot.disconnect()
        print("[System] Shutdown complete.")

    

if __name__ == "__main__":
    main()

