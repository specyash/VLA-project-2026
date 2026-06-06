"""contains the YOLO detection model"""

from __future__ import annotations
import numpy as np
from src.perception.schema import DetectedObject
from src.config import YOLO_DEMO_LABEL, YOLO_CLASS_LABELS, YOLO_CONFIDENCE_THRESHOLD, YOLO_MODEL_PATH, YOLO_IMAGE_SIZE

class YOLODetector:
    """YOLO detection model
        the rest of the codebase only calls detect(frame) using abstraction
        this class contains the entire model for detection"""

    def __init__(self, model_path:str | None = None) -> None:
        self.model_path = model_path or YOLO_MODEL_PATH
        self.model = None

    def detect(self, frame:np.ndarray) -> list[DetectedObject]:

        self._validate_frame(frame)
        model = self._load_model()
        results = model.predict(frame, imgsz=YOLO_IMAGE_SIZE, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

        return self._convert_results(results)

    def _load_model(self):

        if self.model is not None:
            return self.model
        
        if self.model_path is None:
            raise RuntimeError("YOLO model path is not set")

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError("YOLO is not installed. Please install it using 'pip install ultralytics'") from error

        self.model = YOLO(self.model_path)
        return self.model

    def _convert_results(self, results) -> list[DetectedObject]:
        detections: list[DetectedObject] = []
        for result in results:
            class_names = result.names

            for index, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                label = class_names[class_id]
                confidence = float(box.conf[0])
                xyxy = tuple(float(value) for value in box.xyxy[0])
                detection = self._make_detection(detection_id=f"obj_{index}", label=label, confidence=confidence, xyxy_box=xyxy)

                if detection is not None:
                    detections.append(detection)
                
        return detections

    def _parse_labels(self, label:str) -> tuple[str,bool]:

        """parse the label string into a tuple of (label, marked)"""
        
        if label not in YOLO_CLASS_LABELS:
            raise ValueError(f"Invalid YOLO label: {label}")

        parts = label.split("_")
        if len(parts) != 2:
            return ValueError(f"Invalid YOLO label format: {label}")
        
        base_color = parts[0]
        marked_type = parts[1]

        if marked_type not in ["plain", "marked"]:
            raise ValueError(f"Invalid marked type: {marked_type}")
        
        elif marked_type == "plain":
            marked = False

        elif marked_type == "marked":
            marked = True

        return base_color, marked

    def _make_detection(self, detection_id:str, label:str, confidence:float, xyxy_box:tuple[float,float,float,float]) -> DetectedObject | None:


        if confidence < YOLO_CONFIDENCE_THRESHOLD:
            return None

        base_color, marked = self._parse_labels(label)
        x1, y1, x2, y2 = xyxy_box
        x = int(x1)
        y = int(y1)
        width = int(x2 - x)
        height = int(y2 - y)

        if width <= 0 or height <= 0:
            return None

        center_x = x + (width // 2)
        center_y = y + (height // 2)

        area = float(width * height)

        return DetectedObject(id=detection_id, base_color=base_color, marked=marked, label=label, 
        box=(x, y, width, height), center_px=(center_x, center_y), confidence=float(confidence), area=area)

    def detect_demo(self,frame:np.ndarray) -> list[DetectedObject]:
        """demo detection for method testing
            centers of the box and the frame coincide and we assume
            the box width and height to be frame's 1/4th"""

        self._validate_frame(frame)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        box_width = frame_width // 4
        box_height = frame_height // 4

        x1 = (frame_width - box_width) // 2
        y1 = (frame_height - box_height) // 2
        x2 = x1 + box_width
        y2 = y1 + box_height

        demo_detection = self._make_detection(
            detection_id="demo_0",
            label=YOLO_DEMO_LABEL,
            confidence=0.95,
            xyxy_box=(x1,y1,x2,y2)
        )
        if demo_detection is None:
            return []

        return [demo_detection]

    def _validate_frame(self,frame:np.ndarray) -> None:
        if frame is None:
            raise ValueError("YOLO detector received an empty frame")
        
        if not isinstance(frame, np.ndarray):
            raise TypeError("YOLO detector expected a numpy array")

        if frame.ndim != 3:
            raise ValueError("YOLO detector expected a color image with 3 dimensions (height, width, channels)")

        if frame.shape[2] != 3:
            raise ValueError("YOLO detector expected a color image with 3 channels (BGR)")