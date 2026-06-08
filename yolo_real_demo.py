from __future__ import annotations

"""Run real YOLO detection on one image or the webcam.

Usage from generated/:
    python yolo_real_demo.py --image path/to/image.jpg
    python yolo_real_demo.py --webcam
"""



import argparse
from pathlib import Path

import cv2

from src.perception.yolo_detector import YOLODetector


def draw_detection(frame, detection) -> None:
    """Draw one DetectedObject on the image."""

    x, y, width, height = detection.box
    top_left = (x, y)
    bottom_right = (x + width, y + height)
    color = (0, 255, 0)

    cv2.rectangle(frame, top_left, bottom_right, color, 2)
    cv2.circle(frame, detection.center_px, 4, color, -1)

    text = f"{detection.label} {detection.confidence:.2f}"
    cv2.putText(frame, text, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_on_image(image_path: str) -> None:
    """Detect objects in one image file."""

    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = YOLODetector()
    detections = detector.detect(frame)

    print(f"Found {len(detections)} object(s)")
    for detection in detections:
        print(detection)
        draw_detection(frame, detection)

    output_path = Path("data") / "yolo_real_output.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"Saved output to {output_path}")


def run_on_webcam() -> None:
    """Detect objects from the default webcam."""

    detector = YOLODetector()
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press q to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            detections = detector.detect(frame)

            for detection in detections:
                draw_detection(frame, detection)

            cv2.putText(
                frame,
                f"objects: {len(detections)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("YOLO Real Demo", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real YOLO detection demo.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to one input image.")
    group.add_argument("--webcam", action="store_true", help="Use the default webcam.")
    args = parser.parse_args()

    if args.webcam:
        run_on_webcam()
    else:
        run_on_image(args.image)


if __name__ == "__main__":
    main()
