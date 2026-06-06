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
    "yellow_plain",
    "yellow_marked",
)

YOLO_DEMO_LABEL = "blue_plain"
YOLO_CONFIDENCE_THRESHOLD = 0.50