
import numpy as np

def estimate_kabsch_transform(camera_points_mm: np.ndarray, robot_points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the optimal Rigid Transformation (Rotation + Translation) to align
    camera_points_mm with robot_points_mm using the Kabsch algorithm.
    """
    if camera_points_mm.shape != robot_points_mm.shape:
        raise ValueError("Camera and robot point arrays must have the same shape")
    if camera_points_mm.shape[0] < 3:
        raise ValueError("Kabsch needs at least 3 point correspondences")

    camera_center = camera_points_mm.mean(axis=0)
    robot_center = robot_points_mm.mean(axis=0)
    X = camera_points_mm - camera_center
    Y = robot_points_mm - robot_center

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = robot_center - R @ camera_center
    return R, t


def camera_to_robot(camera_point_m: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Converts a single 3D camera coordinate (in meters) into a 3D robot coordinate (in mm)
    using the Kabsch rotation (R) and translation (t).
    """
    camera_point_mm = np.asarray(camera_point_m, dtype=np.float64).reshape(3) * 1000.0
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
