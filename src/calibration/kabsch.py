"""
Kabsch Calibration & Transformation
===================================
Handles the 3D calibration between the camera and the robot using the Kabsch algorithm.
It computes the optimal rotation matrix and translation vector to align 3D camera points
with 3D robot points.
"""

import numpy as np

def estimate_kabsch_transform(camera_points_mm: np.ndarray, robot_points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    The Kabsch algorithm calculates the perfect 3D rotation and movement needed to 
    make the camera's coordinate system match the robot's coordinate system.
    
    Imagine you have a constellation of stars (ArUco markers). The camera sees them 
    from one angle, and the robot arm feels them from a different angle. This function 
    finds the exact mathematical formula to align them perfectly.
    """
    # We need exactly matching lists. For example, point #1 for the camera MUST be point #1 for the robot.
    if camera_points_mm.shape != robot_points_mm.shape:
        raise ValueError("Camera and robot point arrays must have the same shape")
        
    # It takes at least 3 points in 3D space to lock in an alignment (like a tripod)
    if camera_points_mm.shape[0] < 3:
        raise ValueError("Kabsch needs at least 3 point correspondences")

    # Step 1: Find the exact center (average) of all the camera points and robot points
    camera_center = camera_points_mm.mean(axis=0)
    robot_center = robot_points_mm.mean(axis=0)
    
    # Step 2: Move both sets of points so their centers are exactly at 0,0,0
    # This allows us to figure out the rotation without worrying about distance yet
    X = camera_points_mm - camera_center
    Y = robot_points_mm - robot_center

    # Step 3: Calculate the "covariance matrix" (how the points relate to each other)
    H = X.T @ Y
    
    # Step 4: Use Singular Value Decomposition (SVD), a complex math tool that separates 
    # the rotation from the raw data.
    U, _, Vt = np.linalg.svd(H)
    
    # Step 5: Calculate the final 3D Rotation Matrix (R)
    R = Vt.T @ U.T

    # Sometimes math produces a "mirror image" (reflection) instead of a pure rotation.
    # We check the "determinant". If it's negative, it's a mirror image, so we flip it back.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Step 6: Now that we have the rotation, we calculate the exact distance (Translation)
    # needed to move the camera's center to the robot's center.
    t = robot_center - R @ camera_center
    
    # Return Rotation (R) and Translation/Movement (t)
    return R, t


def camera_to_robot(camera_point_m: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Now that we have our mathematical bridge (Rotation R and Translation t),
    this function takes a 3D coordinate from the camera (in meters) and tells us 
    exactly where the robot arm needs to go (in millimeters).
    """
    # Convert camera meters to millimeters so it matches the robot's language
    camera_point_mm = np.asarray(camera_point_m, dtype=np.float64).reshape(3) * 1000.0
    
    # Apply the Rotation (R @ point) and then the Translation (+ t)
    return R @ camera_point_mm + t

def print_calibration_error(camera_points_mm: np.ndarray, robot_points_mm: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Prints and returns the Mean and Max error of the Kabsch calibration."""
    predicted = (R @ camera_points_mm.T).T + t
    errors = np.linalg.norm(predicted - robot_points_mm, axis=1)
    print("Kabsch calibration check:")
    for i, error in enumerate(errors):
        print(f"  point {i}: error={error:.2f} mm")
    print(f"  mean error={errors.mean():.2f} mm, max error={errors.max():.2f} mm")
    return float(errors.mean()), float(errors.max())
