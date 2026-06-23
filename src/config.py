"""Project-wide configuration values.

Runtime constants will live here so perception, planning, calibration, and robot
code do not depend on hardware-specific magic numbers.
"""
YOLO_CLASS_LABELS = (
    "red_plain",
    "red_marked",
    "green_plain",
    "green_marked",
    "blue_plain",
    "blue_marked",
)

YOLO_BASE_CLASS_LABELS = ("blue", "red", "green")

YOLO_DEMO_LABEL = "blue_plain"
YOLO_CONFIDENCE_THRESHOLD = 0.50

YOLO_MODEL_PATH = "YOLO_MODEL/YOLO.pt"
YOLO_IMAGE_SIZE = 640

# Simulated robot pose defaults. These are in millimeters/degrees.
ROBOT_R = 180.0
ROBOT_P = 0.0
ROBOT_YAW = -90.0

# Z heights for pick-and-place planning.
HOVER_Z = 276.0
PICK_Z = 13.0
DROP_Z = 13.0
PICK_Z_OFFSET = 80
DROP_Z_OFFSET = 15.0

# Kabsch 3D Calibration Constants
import numpy as np
import os
import yaml

# Default hardcoded fallbacks
KABSCH_R = np.array([[ 0.97350537,  0.19850607, -0.11350167],
 [ 0.2237692 , -0.92917506,  0.29421259],
 [-0.04705994, -0.31181572, -0.94897646]])
KABSCH_t = np.array([-230.47822236, -610.59690002,  983.93132354])
CAMERA_FX = 606.558349609375
CAMERA_FY = 605.9523315429688
CAMERA_CX = 325.0482482910156
CAMERA_CY = 252.42628479003906
SURFACE_DEPTH = 1006.5

# Load dynamic calibration if file exists
KABSCH_CALIB_FILE = "kabsch_calibration.yaml"
if os.path.exists(KABSCH_CALIB_FILE):
    try:
        with open(KABSCH_CALIB_FILE, "r") as f:
            calib_data = yaml.safe_load(f)
        if calib_data:
            if "KABSCH_R" in calib_data:
                KABSCH_R = np.array(calib_data["KABSCH_R"])
            if "KABSCH_t" in calib_data:
                KABSCH_t = np.array(calib_data["KABSCH_t"])
            CAMERA_FX = calib_data.get("CAMERA_FX", CAMERA_FX)
            CAMERA_FY = calib_data.get("CAMERA_FY", CAMERA_FY)
            CAMERA_CX = calib_data.get("CAMERA_CX", CAMERA_CX)
            CAMERA_CY = calib_data.get("CAMERA_CY", CAMERA_CY)
            SURFACE_DEPTH = calib_data.get("SURFACE_DEPTH", SURFACE_DEPTH)
            PICK_Z = calib_data.get("PICK_Z", PICK_Z)
            DROP_Z = calib_data.get("DROP_Z", DROP_Z)
            print(f"[Config] Dynamic 3D Kabsch calibration loaded from {KABSCH_CALIB_FILE}")
    except Exception as e:
        print(f"[Config Warning] Failed to parse {KABSCH_CALIB_FILE}: {e}. Falling back to default values.")

# Default movement speeds for the fake planner.
DEFAULT_MOVE_SPEED = 50.0
SLOW_MOVE_SPEED = 30.0

# Safe home pose: x, y, z, r, p, yaw.
HOME_POSE = (132.0, 0.0, 174.0, 180.0, -14.0, 0.0)
#HOME_POSE = (201.0, -40.0, 188.0, 180.0, -19, -8)

# Real xArm hardware settings. Coordinate mapping is added later.
ROBOT_IP = "192.168.1.152"
GRIPPER_IP = "192.168.0.147"
MOVE_ACC = 500.0

# Time to wait after gripper commands.
GRIPPER_HTTP_WAIT_SEC = 3.0
GRIPPER_SDK_WAIT_SEC = 3.0

# Collision and safety settings from the old xArm setup.
COLLISION_SENSITIVITY = 3
COLLISION_REBOUND = True
RECOVER_WAIT_SEC = 0.5
RECOVER_MOVE_SPEED = 20.0