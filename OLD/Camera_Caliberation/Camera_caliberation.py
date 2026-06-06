#!/usr/bin/env python3
"""
Final RealSense/OpenCV chessboard calibration script.

What it does:
  1. Optionally captures chessboard images from an Intel RealSense color stream.
  2. Calibrates from the saved images.
  3. Saves camera_calibration.yaml in the format used by cube_pose_estimation_2.py.

Examples:
    python3 calibrate_from_saved_images.py --capture --calibrate
    python3 calibrate_from_saved_images.py --calibrate
    python3 calibrate_from_saved_images.py --capture --width 1280 --height 720 --fps 30
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


DEFAULT_BOARD_SIZE = (7, 5)  # (columns, rows) of inner chessboard corners
DEFAULT_SQUARE_SIZE = 0.047  # meters
DEFAULT_IMAGE_DIR = "calib_images"
DEFAULT_OUTPUT_FILE = "camera_calibration.yaml"


def make_object_points(board_size: tuple[int, int], square_size: float) -> np.ndarray:
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    return objp


def find_chessboard(gray: np.ndarray, board_size: tuple[int, int]):
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            board_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            return True, corners.astype(np.float32)

    found, corners = cv2.findChessboardCorners(
        gray,
        board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        return False, None

    corners_refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return True, corners_refined


def start_realsense(width: int, height: int, fps: int):
    if rs is None:
        sys.exit("pyrealsense2 is required for --capture. Install it or use --calibrate only.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    try:
        profile = pipeline.start(config)
    except RuntimeError as exc:
        sys.exit(f"Cannot start RealSense color stream at {width}x{height}@{fps}: {exc}")

    return pipeline, profile


def capture_images(args) -> None:
    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    pipeline, profile = start_realsense(args.width, args.height, args.fps)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    print("RealSense capture started")
    print(f"Stream: {intr.width}x{intr.height}@{args.fps}")
    print(f"Board: {args.board_cols}x{args.board_rows} inner corners")
    print("Keys: s save detected frame, q quit")

    board_size = (args.board_cols, args.board_rows)
    saved = len(list(image_dir.glob("*.png"))) if args.continue_numbering else 0

    cv2.namedWindow("Calibration Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Capture", args.width, args.height)

    try:
        while saved < args.num_images:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_chessboard(gray, board_size)

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, board_size, corners, found)
                status = "Board detected - press s"
                color = (0, 255, 0)
            else:
                status = "No board detected"
                color = (0, 0, 255)

            cv2.putText(display, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(
                display,
                f"Saved: {saved}/{args.num_images}",
                (12, display.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )
            cv2.imshow("Calibration Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if not found:
                    print("Skipped: chessboard not detected")
                    continue
                filename = image_dir / f"calib_{saved:03d}.png"
                cv2.imwrite(str(filename), frame)
                saved += 1
                print(f"Saved {filename}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"Capture complete: {saved} images in {image_dir}")


def image_paths(image_dir: str) -> list[str]:
    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(image_dir, pattern)))
    return sorted(paths)


def reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist) -> float:
    total_error = 0.0
    for i, objp in enumerate(objpoints):
        projected, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
        total_error += cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
    return total_error / len(objpoints)


def write_yaml(
    path: str,
    K: np.ndarray,
    dist: np.ndarray,
    image_size: tuple[int, int],
    board_size: tuple[int, int],
    square_size: float,
    rms_error: float,
    mean_error: float,
) -> None:
    dist_flat = dist.reshape(1, -1)
    lines = [
        "camera_matrix:",
        "   rows: 3",
        "   cols: 3",
        "   dt: d",
        "   data: [ " + ", ".join(f"{v:.16g}" for v in K.reshape(-1)) + " ]",
        "dist_coeffs:",
        f"   rows: {dist_flat.shape[0]}",
        f"   cols: {dist_flat.shape[1]}",
        "   dt: d",
        "   data: [ " + ", ".join(f"{v:.16g}" for v in dist_flat.reshape(-1)) + " ]",
        f"image_width: {image_size[0]}",
        f"image_height: {image_size[1]}",
        f"board_cols: {board_size[0]}",
        f"board_rows: {board_size[1]}",
        f"square_size: {square_size:.16g}",
        f"rms_error: {rms_error:.16g}",
        f"mean_reprojection_error: {mean_error:.16g}",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def calibrate_from_images(args) -> None:
    board_size = (args.board_cols, args.board_rows)
    objp = make_object_points(board_size, args.square_size)
    images = image_paths(args.image_dir)

    if len(images) < args.min_images:
        sys.exit(f"Need at least {args.min_images} images in {args.image_dir}. Found {len(images)}.")

    objpoints = []
    imgpoints = []
    image_size = None

    for filename in images:
        img = cv2.imread(filename)
        if img is None:
            print(f"Skipped unreadable image: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = find_chessboard(gray, board_size)
        if not found:
            print(f"Skipped, no corners: {filename}")
            continue

        if image_size is None:
            image_size = gray.shape[::-1]
        elif image_size != gray.shape[::-1]:
            print(f"Skipped, different size: {filename}")
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        print(f"Using: {filename}")

    if len(objpoints) < args.min_images:
        sys.exit(f"Need at least {args.min_images} valid detections. Found {len(objpoints)}.")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    mean_error = reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    write_yaml(
        args.output,
        K,
        dist,
        image_size,
        board_size,
        args.square_size,
        rms,
        mean_error,
    )

    print("")
    print("Calibration complete")
    print(f"Valid images: {len(objpoints)}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"RMS error: {rms:.6f}")
    print(f"Mean reprojection error: {mean_error:.6f}")
    print(f"Saved YAML: {args.output}")

    if args.show_undistort:
        preview = cv2.imread(images[0])
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, image_size, 1, image_size)
        undistorted = cv2.undistort(preview, K, dist, None, new_K)
        cv2.imshow("Original", preview)
        cv2.imshow("Undistorted", undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Capture and calibrate RealSense chessboard images.")
    parser.add_argument("--capture", action="store_true", help="Capture calibration images from RealSense")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate from saved images and save YAML")
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--board-cols", type=int, default=DEFAULT_BOARD_SIZE[0])
    parser.add_argument("--board-rows", type=int, default=DEFAULT_BOARD_SIZE[1])
    parser.add_argument("--square-size", type=float, default=DEFAULT_SQUARE_SIZE, help="Chessboard square size in meters")
    parser.add_argument("--min-images", type=int, default=10)
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--continue-numbering", action="store_true")
    parser.add_argument("--show-undistort", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.capture and not args.calibrate:
        args.calibrate = True

    if args.capture:
        capture_images(args)
    if args.calibrate:
        calibrate_from_images(args)


if __name__ == "__main__":
    main()
