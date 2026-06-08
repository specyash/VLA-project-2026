"""
ArUco 3D Pose Estimation
========================
Handles the 3D pose estimation (Position & Rotation) of ArUco markers 
in real-world space using camera calibration parameters.
"""

import cv2
import numpy as np


def marker_object_points(marker_size: float) -> np.ndarray:
    """
    Returns the 3D coordinates of the 4 corners of the marker in its own local coordinate system.
    Used for solvePnP pose estimation.
    """
    half = marker_size / 2.0
    return np.array([
        [-half,  half, 0.0],  # Top-Left corner
        [ half,  half, 0.0],  # Top-Right corner
        [ half, -half, 0.0],  # Bottom-Right corner
        [-half, -half, 0.0],  # Bottom-Left corner
    ], dtype=np.float32)


def estimate_marker_pose(corners, marker_size: float, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimates the 3D translation (tvecs) and rotation (rvecs) of ArUco markers.
    Works for both older OpenCV (cv2.aruco.estimatePoseSingleMarkers) and newer OpenCV (cv2.solvePnP).
    """
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, K, dist)
        if rvecs is None: return np.array([]), np.array([])
        return rvecs.reshape(-1, 3), tvecs.reshape(-1, 3)

    # Fallback to solvePnP for newer OpenCV versions
    objp = marker_object_points(marker_size)
    rvecs = []
    tvecs = []
    
    for marker_corners in corners:
        ok, rvec, tvec = cv2.solvePnP(objp, marker_corners.reshape(4, 2), K, dist)
        if ok:
            rvecs.append(rvec.reshape(3))
            tvecs.append(tvec.reshape(3))
        else:
            rvecs.append(np.full(3, np.nan))
            tvecs.append(np.full(3, np.nan))
            
    return np.array(rvecs), np.array(tvecs)


def detect_target_pose(frame: np.ndarray, detector, target_id: int, marker_size: float, K: np.ndarray, dist: np.ndarray):
    """
    Detects markers in the frame, draws them, and returns the 3D pose of a specific target_id.
    Returns: tvec, rvec, all_corners, all_ids
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Handle older vs newer OpenCV API differences
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        dictionary, params = detector
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
        
    if ids is None or len(ids) == 0:
        return None, None, corners, ids

    # Draw standard bounding boxes
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    # Calculate 3D Positional Math
    rvecs, tvecs = estimate_marker_pose(corners, marker_size, K, dist)

    # Search for the exact target ID
    for i, marker_id in enumerate(ids.flatten()):
        if int(marker_id) != target_id:
            continue
            
        if np.isnan(tvecs[i]).any():
            return None, None, corners, ids
            
        # Draw 3D axis directly onto the live frame
        cv2.drawFrameAxes(frame, K, dist, rvecs[i], tvecs[i], marker_size * 0.5)
        
        return tvecs[i], rvecs[i], corners, ids

    return None, None, corners, ids
