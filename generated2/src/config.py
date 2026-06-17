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
ROBOT_R = -180.0
ROBOT_P = 0.0
ROBOT_YAW = -90.0

# Z heights for pick-and-place planning.
HOVER_Z = 276.0
PICK_Z = 176.0
DROP_Z = 176.0

# Default movement speeds for the fake planner.
DEFAULT_MOVE_SPEED = 20.0
SLOW_MOVE_SPEED = 10.0

# Safe home pose: x, y, z, r, p, yaw.
HOME_POSE = (0.0, -250.0, 300.0, ROBOT_R, ROBOT_P, ROBOT_YAW)

# Real xArm hardware settings. Coordinate mapping is added later.
ROBOT_IP = "192.168.1.152"
GRIPPER_IP = "192.168.0.147"
MOVE_ACC = 500.0

# Time to wait after gripper commands.
GRIPPER_HTTP_WAIT_SEC = 3.0
GRIPPER_SDK_WAIT_SEC = 1.0

# Collision and safety settings from the old xArm setup.
COLLISION_SENSITIVITY = 3
COLLISION_REBOUND = True
RECOVER_WAIT_SEC = 0.5
RECOVER_MOVE_SPEED = 20.0