"""Tests for the YOLO detector helper methods."""

import numpy as np
import pytest

from src.perception.yolo_detector import YOLODetector


def test_parse_base_colors_accepts_known_color() -> None:
    """A plain label should return its color and marked=False."""

    detector = YOLODetector()
    assert detector._parse_base_color("blue") == "blue"
    assert detector._parse_base_color("green") == "green"
    assert detector._parse_base_color("red") == "red"

def test_parse_base_color_rejects_unknown_color() -> None:
    """Unknown YOLO colors should fail clearly."""

    detector = YOLODetector()

    with pytest.raises(ValueError, match="Unknown YOLO base color"):
        detector._parse_base_color("purple")

def test_compose_plain_label() -> None:
    """Base color plus marked=False should become color_plain."""

    detector = YOLODetector()

    assert detector._compose_label("blue", marked=False) == "blue_plain"

def test_compose_marked_label() -> None:
    """Base color plus marked=True should become color_marked."""

    detector = YOLODetector()

    assert detector._compose_label("red", marked=True) == "red_marked"

def test_parse_plain_label() -> None:
    """A full plain label should split correctly."""

    detector = YOLODetector()

    base_color, marked = detector._parse_labels("blue_plain")

    assert base_color == "blue"
    assert marked is False

def test_parse_marked_label() -> None:
    """A marked label should return its color and marked=True."""

    detector = YOLODetector()

    base_color, marked = detector._parse_labels("red_marked")

    assert base_color == "red"
    assert marked is True


def test_parse_unknown_label_raises_error() -> None:
    """Labels outside the configured class list should fail clearly."""

    detector = YOLODetector()

    with pytest.raises(ValueError):
        detector._parse_labels("purple_plain")


def test_make_detection_converts_box_format() -> None:
    """YOLO xyxy boxes should become OpenCV x, y, width, height boxes."""

    detector = YOLODetector()

    detection = detector._make_detection(
        detection_id="obj_0",
        label="green_marked",
        confidence=0.95,
        xyxy_box=(10, 20, 50, 80),
    )

    assert detection is not None
    assert detection.id == "obj_0"
    assert detection.base_color == "green"
    assert detection.marked is True
    assert detection.label == "green_marked"
    assert detection.box == (10, 20, 40, 60)
    assert detection.center_px == (30, 50)
    assert detection.confidence == 0.95
    assert detection.area == 2400.0


def test_make_detection_skips_low_confidence() -> None:
    """Low-confidence detections should be ignored."""

    detector = YOLODetector()

    detection = detector._make_detection(
        detection_id="obj_0",
        label="blue_plain",
        confidence=0.10,
        xyxy_box=(10, 20, 50, 80),
    )

    assert detection is None


def test_make_detection_skips_invalid_box() -> None:
    """Boxes with no positive width or height should be ignored."""

    detector = YOLODetector()

    detection = detector._make_detection(
        detection_id="obj_0",
        label="blue_plain",
        confidence=0.95,
        xyxy_box=(50, 20, 10, 80),
    )

    assert detection is None


def test_detect_demo_returns_one_detection() -> None:
    """The demo detector should produce one fake detection for a valid frame."""

    detector = YOLODetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect_demo(frame)

    assert len(detections) == 1
    assert detections[0].label == "blue_plain"

def test_detect_requires_model_path_for_real_yolo(monkeypatch) -> None:
    """Real detection should fail clearly until a model path is configured."""

    monkeypatch.setattr("src.perception.yolo_detector.YOLO_MODEL_PATH", None)
    detector = YOLODetector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="YOLO model path"):
        detector.detect(frame)

