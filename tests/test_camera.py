"""Tests for the Camera class, covering both RealSense functionality and webcam fallback."""

import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import src.camera.camera as camera_module
from src.camera.camera import Camera


@pytest.fixture
def mock_cv(monkeypatch):
    """Fixture to mock cv2 (imported as cv in camera.py)."""
    mock_cv_mod = MagicMock()
    mock_capture = MagicMock()
    mock_cv_mod.VideoCapture.return_value = mock_capture
    monkeypatch.setattr(camera_module, "cv", mock_cv_mod)
    return mock_cv_mod, mock_capture


@pytest.fixture
def mock_realsense(monkeypatch):
    """Fixture to mock pyrealsense2 (mocked as prs in camera.py)."""
    mock_prs = MagicMock()
    mock_pipeline = MagicMock()
    mock_config = MagicMock()
    mock_align = MagicMock()

    mock_prs.pipeline.return_value = mock_pipeline
    mock_prs.config.return_value = mock_config
    mock_prs.align.return_value = mock_align

    # Bind streams
    mock_prs.stream.color = "color_stream"
    mock_prs.stream.depth = "depth_stream"
    mock_prs.format.bgr8 = "bgr8"
    mock_prs.format.z16 = "z16"

    monkeypatch.setattr(camera_module, "prs", mock_prs, raising=False)
    return mock_prs, mock_pipeline, mock_config, mock_align


def test_realsense_init_success(monkeypatch, mock_realsense, mock_cv) -> None:
    """If prs_ready is True, Camera should initialize the RealSense pipeline successfully."""
    monkeypatch.setattr(camera_module, "prs_ready", True)
    _, mock_capture = mock_cv

    cam = Camera(width=640, height=480, fps=15)

    assert cam._using_prs is True
    assert cam.has_depth() is True
    assert cam._pipeline is not None
    assert cam._align is not None
    assert cam.width == 640
    assert cam.height == 480
    assert cam.fps == 15

    # OpenCV Capture should not have been initialized
    mock_cv[0].VideoCapture.assert_not_called()


def test_realsense_read_success(monkeypatch, mock_realsense) -> None:
    """RealSense read() should process aligned frames and return numpy arrays."""
    monkeypatch.setattr(camera_module, "prs_ready", True)
    _, mock_pipeline, _, mock_align = mock_realsense

    # Setup mock frames returned by pipeline
    mock_frames = MagicMock()
    mock_aligned_frames = MagicMock()
    mock_color_frame = MagicMock()
    mock_depth_frame = MagicMock()

    mock_pipeline.wait_for_frames.return_value = mock_frames
    mock_align.process.return_value = mock_aligned_frames
    mock_aligned_frames.get_color_frame.return_value = mock_color_frame
    mock_aligned_frames.get_depth_frame.return_value = mock_depth_frame

    # Set mock numpy array data return values
    color_data = np.zeros((480, 640, 3), dtype=np.uint8)
    depth_data = np.ones((480, 640), dtype=np.uint16)

    mock_color_frame.get_data.return_value = color_data
    mock_depth_frame.get_data.return_value = depth_data

    cam = Camera()
    ok, color, depth = cam.read()

    assert ok is True
    assert np.array_equal(color, color_data)
    assert np.array_equal(depth, depth_data)


def test_realsense_read_missing_frames(monkeypatch, mock_realsense) -> None:
    """If color or depth frames are missing from wait_for_frames, read() should return False, None, None."""
    monkeypatch.setattr(camera_module, "prs_ready", True)
    _, mock_pipeline, _, mock_align = mock_realsense

    mock_frames = MagicMock()
    mock_aligned_frames = MagicMock()

    mock_pipeline.wait_for_frames.return_value = mock_frames
    mock_align.process.return_value = mock_aligned_frames
    
    # Simulate missing color frame
    mock_aligned_frames.get_color_frame.return_value = None
    mock_aligned_frames.get_depth_frame.return_value = MagicMock()

    cam = Camera()
    ok, color, depth = cam.read()

    assert ok is False
    assert color is None
    assert depth is None


def test_realsense_read_exception_returns_false(monkeypatch, mock_realsense) -> None:
    """If wait_for_frames raises an exception, read() should catch it and return False, None, None."""
    monkeypatch.setattr(camera_module, "prs_ready", True)
    _, mock_pipeline, _, _ = mock_realsense

    mock_pipeline.wait_for_frames.side_effect = Exception("Pipeline error")

    cam = Camera()
    ok, color, depth = cam.read()

    assert ok is False
    assert color is None
    assert depth is None


def test_realsense_release(monkeypatch, mock_realsense) -> None:
    """Releasing a RealSense camera stops the pipeline."""
    monkeypatch.setattr(camera_module, "prs_ready", True)
    _, mock_pipeline, _, _ = mock_realsense

    cam = Camera()
    cam.release()

    mock_pipeline.stop.assert_called_once()


def test_webcam_fallback_when_realsense_fails(monkeypatch, mock_realsense, mock_cv) -> None:
    """If RealSense pipeline start raises an exception, Camera should fallback to OpenCV webcam."""
    monkeypatch.setattr(camera_module, "prs_ready", True)
    _, mock_pipeline, _, _ = mock_realsense
    mock_cv_mod, mock_capture = mock_cv

    # Force pipeline.start() to raise an exception
    mock_pipeline.start.side_effect = RuntimeError("No RealSense device connected")

    cam = Camera(width=800, height=600)

    assert cam._using_prs is False
    assert cam.has_depth() is False
    assert cam._pipeline is None
    assert cam._capture is not None

    # OpenCV VideoCapture(0) should be initialized and properties set
    mock_cv_mod.VideoCapture.assert_called_once_with(0)
    mock_capture.set.assert_any_call(mock_cv_mod.CAP_PROP_FRAME_WIDTH, 800)
    mock_capture.set.assert_any_call(mock_cv_mod.CAP_PROP_FRAME_HEIGHT, 600)


def test_webcam_fallback_when_realsense_not_ready(monkeypatch, mock_cv) -> None:
    """If prs_ready is False, Camera should initialize the OpenCV webcam fallback directly."""
    monkeypatch.setattr(camera_module, "prs_ready", False)
    mock_cv_mod, mock_capture = mock_cv

    cam = Camera(width=1280, height=720)

    assert cam._using_prs is False
    assert cam.has_depth() is False
    assert cam._capture is not None

    # OpenCV VideoCapture(0) should be initialized and properties set
    mock_cv_mod.VideoCapture.assert_called_once_with(0)
    mock_capture.set.assert_any_call(mock_cv_mod.CAP_PROP_FRAME_WIDTH, 1280)
    mock_capture.set.assert_any_call(mock_cv_mod.CAP_PROP_FRAME_HEIGHT, 720)


def test_webcam_read_success(monkeypatch, mock_cv) -> None:
    """Webcam read() should return the BGR frame from VideoCapture.read() and None for depth."""
    monkeypatch.setattr(camera_module, "prs_ready", False)
    _, mock_capture = mock_cv

    frame_data = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_capture.read.return_value = (True, frame_data)

    cam = Camera()
    ok, color, depth = cam.read()

    assert ok is True
    assert np.array_equal(color, frame_data)
    assert depth is None


def test_webcam_release(monkeypatch, mock_cv) -> None:
    """Releasing a webcam releases the VideoCapture resource."""
    monkeypatch.setattr(camera_module, "prs_ready", False)
    _, mock_capture = mock_cv

    cam = Camera()
    cam.release()

    mock_capture.release.assert_called_once()
