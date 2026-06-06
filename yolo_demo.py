import numpy as np
from src.perception.yolo_detector import YOLODetector
import cv2 as cv

def main() -> None:
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = YOLODetector()
    detections = detector.detect_demo(frame)

    for detection in detections:
        print(detection)
        draw_detection(frame, detection)

    output_path = "data/yolo_demo_output.jpg"
    cv.imwrite(output_path, frame)
    print(f"Demo output saved to {output_path}")

def draw_detection(frame:np.ndarray, detection) -> np.ndarray:
    """draw the detection on the frame"""
    x, y, width, height = detection.box
    center_x, center_y = detection.center_px

    top_left = (x, y)
    bottom_right = (x + width, y + height)

    box_color = (0,255,0)

    cv.rectangle(frame, top_left, bottom_right, box_color,thickness=2)
    cv.circle(frame, (center_x, center_y), radius=4, color=box_color, thickness=-1)

    text = f"{detection.label} {detection.confidence:.2f}"
    text_position = (x, y - 10)
    cv.putText(frame, text, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, box_color, thickness=2)

if __name__ == "__main__":
    main()