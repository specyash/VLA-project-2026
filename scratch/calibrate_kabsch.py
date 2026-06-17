import cv2 as cv
import numpy as np
import pyrealsense2 as prs
from src.camera.camera import Camera
from src.calibration.workspace import make_aruco_detector, detect_tags, ROBOT_READINGS, CORNER_ORDER, TAG_TO_INDEX
from src.calibration.kabsch import estimate_kabsch_transform, print_calibration_error

def main():
    print("[Calibration] Initializing Camera...")
    camera = Camera()
    
    if not camera._using_prs:
        print("[Error] Kabsch 3D calibration requires a RealSense depth camera.")
        camera.release()
        return

    # 1. Retrieve Intrinsic Parameters from RealSense pipeline
    try:
        profile = camera._pipeline.get_active_profile()
        color_stream = profile.get_stream(prs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        fx = intrinsics.fx
        fy = intrinsics.fy
        cx = intrinsics.ppx
        cy = intrinsics.ppy
        print(f"[Calibration] Color Intrinsics loaded: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    except Exception as e:
        print(f"[Error] Failed to read RealSense intrinsics: {e}")
        camera.release()
        return

    # Prompt user for table height Z
    table_z_input = input("Enter the table Z coordinate (in mm, e.g. 100): ")
    try:
        table_z = float(table_z_input)
    except ValueError:
        table_z = 100.0
    print(f"Using Table Z = {table_z} mm")

    print("\nDetecting AprilTags. Press 'c' in the video window once tags are stable to compute calibration...")
    detector = make_aruco_detector()
    
    stable_workspace = {}
    
    try:
        while True:
            ok, frame, depth_frame = camera.read()
            if not ok or frame is None or depth_frame is None:
                continue
                
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            found_ws, _ = detect_tags(gray, detector)
            
            # Draw tag centers on preview
            preview = frame.copy()
            for tag_id, (tx, ty) in found_ws.items():
                depth_val = depth_frame[ty, tx]
                cv.circle(preview, (tx, ty), 6, (0, 255, 0), -1)
                cv.putText(preview, f"ID {tag_id} ({depth_val}mm)", (tx + 10, ty - 10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show number of detected tags
            cv.putText(preview, f"Workspace Tags Detected: {len(found_ws)}/4 (Need all 4)", 
                       (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if len(found_ws) < 4 else (0, 255, 0), 2)
            cv.putText(preview, "Press 'c' to Calibrate, 'q' to Quit", 
                       (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                       
            cv.imshow("Calibration Preview", preview)
            
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Calibration cancelled.")
                break
            elif key == ord('c'):
                if len(found_ws) == 4:
                    stable_workspace = found_ws.copy()
                    print("Locked 4 workspace tag centers!")
                    break
                else:
                    print(f"Cannot calibrate. Only detected {len(found_ws)} tags.")
    finally:
        camera.release()
        cv.destroyAllWindows()

    if len(stable_workspace) != 4:
        return

    # 2. Build Camera 3D points (in mm) and Robot 3D points (in mm)
    cam_pts_list = []
    rob_pts_list = []
    
    print("\n--- Point Correspondences ---")
    for tag_id in CORNER_ORDER:
        u, v = stable_workspace[tag_id]
        d_mm = float(depth_frame[v, u])
        
        # Project 2D pixel + depth to 3D camera coordinates (in mm)
        xc = ((u - cx) * d_mm) / fx
        yc = ((v - cy) * d_mm) / fy
        zc = d_mm
        
        # Robot ground truth position from ROBOT_READINGS config + table height Z
        idx = TAG_TO_INDEX[tag_id]
        xr, yr = ROBOT_READINGS[idx]
        zr = table_z
        
        print(f"Tag ID {tag_id}:")
        print(f"  Camera 3D (X, Y, Z): ({xc:.2f}, {yc:.2f}, {zc:.2f}) mm")
        print(f"  Robot 3D  (X, Y, Z): ({xr:.2f}, {yr:.2f}, {zr:.2f}) mm")
        
        cam_pts_list.append([xc, yc, zc])
        rob_pts_list.append([xr, yr, zr])
        
    cam_pts = np.array(cam_pts_list, dtype=np.float64)
    rob_pts = np.array(rob_pts_list, dtype=np.float64)
    
    # 3. Compute optimal Rotation (R) and Translation (t) using Kabsch
    R, t = estimate_kabsch_transform(cam_pts, rob_pts)
    
    print("\n--- Calibration Results ---")
    print("Rotation Matrix (R):")
    print(R)
    print("\nTranslation Vector (t):")
    print(t)
    print("----------------------------\n")
    
    # Check error
    print_calibration_error(cam_pts, rob_pts, R, t)
    
    # Save calibration to YAML file
    import yaml
    surface_depth = float(cam_pts[:, 2].mean())
    calib_data = {
        'KABSCH_R': R.tolist(),
        'KABSCH_t': t.tolist(),
        'CAMERA_FX': float(fx),
        'CAMERA_FY': float(fy),
        'CAMERA_CX': float(cx),
        'CAMERA_CY': float(cy),
        'SURFACE_DEPTH': float(surface_depth),
        'PICK_Z': float(table_z),
        'DROP_Z': float(table_z)
    }
    calib_filename = "kabsch_calibration.yaml"
    try:
        with open(calib_filename, "w") as f:
            yaml.safe_dump(calib_data, f)
        print(f"\n[Success] Calibration saved automatically to {calib_filename}!")
    except Exception as e:
        print(f"\n[Error] Failed to save calibration to YAML: {e}")

    # Generate copy-paste python configuration code
    print("\nOr, add the following constants to your configuration manually:")
    print("=" * 60)
    print(f"KABSCH_R = np.array({np.array2string(R, separator=', ')})")
    print(f"KABSCH_t = np.array({np.array2string(t, separator=', ')})")
    print(f"CAMERA_FX = {fx}")
    print(f"CAMERA_FY = {fy}")
    print(f"CAMERA_CX = {cx}")
    print(f"CAMERA_CY = {cy}")
    print(f"SURFACE_DEPTH = {surface_depth}")
    print("=" * 60)

if __name__ == "__main__":
    main()
