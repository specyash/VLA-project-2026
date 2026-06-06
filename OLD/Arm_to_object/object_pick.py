#!/usr/bin/env python3
"""
Detect an object AprilTag, transform its camera pose into the Lite6 robot base
frame with Kabsch calibration, and move the arm above the object.

Default calibration correspondences:
    workspace tag 16 -> robot saved_readings[0]
    workspace tag 17 -> robot saved_readings[1]
    workspace tag 18 -> robot saved_readings[2]
    workspace tag 19 -> robot saved_readings[3]

If your third workspace tag is really ID 8, run with:
    --tag-map 16:0,17:1,8:2,19:3
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


ROBOT_IP = "192.168.1.152"

ROBOT_R = -178.0
ROBOT_P = 0.0
ROBOT_Y = -88.0

APPROACH_Z_OFFSET_MM = 100.0
MOVE_SPEED = 40
MOVE_ACC = 500

TARGET_TAG_ID = 0
TAG_SIZE_M = 0.05

APRILTAG_DICTS = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def parse_data_block(text: str, key: str, expected: int | None = None) -> list[float]:
    pattern = rf"{re.escape(key)}:\s*(?:\n(?:[^\n]*\n)*?\s*data:\s*)?\[([^\]]+)\]"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Cannot find '{key}' data in calibration YAML")

    values = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))]
    if expected is not None and len(values) != expected:
        raise ValueError(f"Expected {expected} values for '{key}', got {len(values)}")
    return values


def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:
    text = Path(path).read_text()
    K = np.array(parse_data_block(text, "camera_matrix", 9), dtype=np.float64).reshape(3, 3)
    dist = np.array(parse_data_block(text, "dist_coeffs"), dtype=np.float64).reshape(1, -1)
    return K, dist


def load_workspace_tags(path: str) -> dict[int, np.ndarray]:
    text = Path(path).read_text()
    tags: dict[int, np.ndarray] = {}
    block_pattern = re.compile(
        r"^\s{4}(\d+):\s*\n"
        r"(?:^\s{6}.+\n)*?"
        r"^\s{6}x:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\n"
        r"^\s{6}y:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\n"
        r"^\s{6}z:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*$",
        flags=re.MULTILINE,
    )

    for match in block_pattern.finditer(text):
        tag_id = int(match.group(1))
        tags[tag_id] = np.array(
            [float(match.group(2)), float(match.group(3)), float(match.group(4))],
            dtype=np.float64,
        )

    if not tags:
        raise ValueError(f"No tag positions found in {path}")
    return tags


def load_robot_saved_readings(path: str) -> list[np.ndarray]:
    text = Path(path).read_text()
    readings: list[np.ndarray] = []
    block_pattern = re.compile(
        r"^\s{4}(\d+):\s*\n"
        r"^\s{6}x:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\n"
        r"^\s{6}y:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*\n"
        r"^\s{6}z:\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)\s*",
        flags=re.MULTILINE,
    )

    indexed: dict[int, np.ndarray] = {}
    for match in block_pattern.finditer(text):
        index = int(match.group(1))
        indexed[index] = np.array(
            [float(match.group(2)), float(match.group(3)), float(match.group(4))],
            dtype=np.float64,
        )

    if not indexed:
        raise ValueError(f"No saved robot readings found in {path}")

    for index in sorted(indexed):
        readings.append(indexed[index])
    return readings


def parse_tag_map(text: str) -> dict[int, int]:
    tag_map: dict[int, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        tag_text, index_text = item.split(":")
        tag_map[int(tag_text.strip())] = int(index_text.strip())
    return tag_map


def estimate_kabsch_transform(camera_points_mm: np.ndarray, robot_points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = robot_center - R @ camera_center
    return R, t


def camera_to_robot(camera_point_m: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    camera_point_mm = np.asarray(camera_point_m, dtype=np.float64).reshape(3) * 1000.0
    return R @ camera_point_mm + t


def build_transform(
    workspace_tags: dict[int, np.ndarray],
    robot_readings: list[np.ndarray],
    tag_map: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = []
    robot_points = []

    for tag_id, reading_index in tag_map.items():
        if tag_id not in workspace_tags:
            raise ValueError(f"Workspace YAML does not contain tag {tag_id}")
        if reading_index >= len(robot_readings):
            raise ValueError(f"Robot YAML does not contain saved_readings[{reading_index}]")

        camera_points.append(workspace_tags[tag_id] * 1000.0)
        robot_points.append(robot_readings[reading_index])

    camera_points_mm = np.vstack(camera_points)
    robot_points_mm = np.vstack(robot_points)
    R, t = estimate_kabsch_transform(camera_points_mm, robot_points_mm)
    return R, t, robot_points_mm


def print_calibration_error(
    camera_points_mm: np.ndarray,
    robot_points_mm: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[float, float]:
    predicted = (R @ camera_points_mm.T).T + t
    errors = np.linalg.norm(predicted - robot_points_mm, axis=1)
    print("Kabsch calibration check:")
    for i, error in enumerate(errors):
        print(f"  point {i}: error={error:.2f} mm")
    print(f"  mean error={errors.mean():.2f} mm, max error={errors.max():.2f} mm")
    return float(errors.mean()), float(errors.max())


def write_tag_robot_mapping_yaml(
    path: str,
    workspace_tags: dict[int, np.ndarray],
    robot_readings: list[np.ndarray],
    tag_map: dict[int, int],
    R: np.ndarray,
    t: np.ndarray,
    mean_error_mm: float,
    max_error_mm: float,
) -> None:
    lines = [
        "tag_robot_mapping:",
        f"  created_unix_time: {time.time():.3f}",
        "  camera_frame_units: m",
        "  robot_frame_units: mm",
        f"  mean_calibration_error_mm: {mean_error_mm:.8f}",
        f"  max_calibration_error_mm: {max_error_mm:.8f}",
        "  correspondences:",
    ]

    for tag_id in sorted(tag_map):
        reading_index = tag_map[tag_id]
        cam_xyz_m = workspace_tags[tag_id]
        robot_xyz_mm = robot_readings[reading_index]
        lines.extend(
            [
                f"    - tag_id: {tag_id}",
                f"      robot_saved_reading_index: {reading_index}",
                f"      camera_xyz_m: [{cam_xyz_m[0]:.8f}, {cam_xyz_m[1]:.8f}, {cam_xyz_m[2]:.8f}]",
                f"      robot_xyz_mm: [{robot_xyz_mm[0]:.8f}, {robot_xyz_mm[1]:.8f}, {robot_xyz_mm[2]:.8f}]",
            ]
        )

    lines.extend(
        [
            "  transform_camera_to_robot:",
            "    rotation_matrix:",
            f"      - [{R[0,0]:.12f}, {R[0,1]:.12f}, {R[0,2]:.12f}]",
            f"      - [{R[1,0]:.12f}, {R[1,1]:.12f}, {R[1,2]:.12f}]",
            f"      - [{R[2,0]:.12f}, {R[2,1]:.12f}, {R[2,2]:.12f}]",
            f"    translation_mm: [{t[0]:.8f}, {t[1]:.8f}, {t[2]:.8f}]",
        ]
    )

    Path(path).write_text("\n".join(lines) + "\n")


def write_object_pose_yaml(
    path: str,
    target_tag: int,
    camera_xyz_m: np.ndarray,
    robot_xyz_mm: np.ndarray,
    inside_workspace_flag: bool,
) -> None:
    lines = [
        "object_pose:",
        f"  created_unix_time: {time.time():.3f}",
        f"  target_tag: {target_tag}",
        "  camera_frame_units: m",
        "  robot_frame_units: mm",
        f"  inside_workspace: {str(bool(inside_workspace_flag)).lower()}",
        (
            "  camera_xyz_m: "
            f"[{camera_xyz_m[0]:.8f}, {camera_xyz_m[1]:.8f}, {camera_xyz_m[2]:.8f}]"
        ),
        (
            "  robot_xyz_mm: "
            f"[{robot_xyz_mm[0]:.8f}, {robot_xyz_mm[1]:.8f}, {robot_xyz_mm[2]:.8f}]"
        ),
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def make_detector(dict_name: str):
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTS[dict_name])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return dictionary, params


def detect_markers(detector, gray: np.ndarray):
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, params = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids, rejected


def marker_object_points(marker_size: float) -> np.ndarray:
    half = marker_size / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def estimate_marker_pose(corners, marker_size: float, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(cv2.aruco, "estimatePoseSingleMarkers"):
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, K, dist)
        return rvecs.reshape(-1, 3), tvecs.reshape(-1, 3)

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


class RealSenseSource:
    def __init__(self, width: int, height: int, fps: int) -> None:
        if rs is None:
            sys.exit("pyrealsense2 is required for --realsense. Use --camera instead.")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline.start(config)

    def read(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return False, None
        return True, np.asanyarray(color_frame.get_data())

    def release(self) -> None:
        self.pipeline.stop()


def open_source(args: argparse.Namespace):
    if args.realsense:
        return RealSenseSource(args.width, args.height, args.fps)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    return cap


def inside_workspace(robot_xyz: np.ndarray, workspace_robot_points: np.ndarray) -> bool:
    polygon = workspace_robot_points[:, :2].astype(np.float32)
    hull = cv2.convexHull(polygon)
    distance = cv2.pointPolygonTest(hull, (float(robot_xyz[0]), float(robot_xyz[1])), False)
    return distance >= 0


def inside_xy_limits(robot_xyz: np.ndarray, limits: tuple[float, float, float, float]) -> bool:
    min_x, max_x, min_y, max_y = limits
    return min_x <= float(robot_xyz[0]) <= max_x and min_y <= float(robot_xyz[1]) <= max_y


def inside_hover_z_limits(hover_z: float, args: argparse.Namespace) -> bool:
    return args.min_hover_z <= hover_z <= args.max_hover_z


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def resolve_xy_limits(args: argparse.Namespace, workspace_robot_points: np.ndarray) -> tuple[float, float, float, float]:
    if not args.auto_xy_limits:
        return args.min_x, args.max_x, args.min_y, args.max_y

    xs = workspace_robot_points[:, 0]
    ys = workspace_robot_points[:, 1]
    min_x = float(xs.min() - args.xy_margin)
    max_x = float(xs.max() + args.xy_margin)
    min_y = float(ys.min() - args.xy_margin)
    max_y = float(ys.max() + args.xy_margin)
    return min_x, max_x, min_y, max_y


def check_code(name: str, code: int) -> None:
    if code != 0:
        print(f"Warning: {name} returned xArm code {code}")


def safe_disconnect(arm: Any | None) -> None:
    if arm is None:
        return
    try:
        arm.disconnect()
    except Exception as exc:
        print(f"Warning: robot disconnect failed: {exc}")


def connect_arm(ip: str):
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        sys.exit("xArm Python SDK not found. Install it with: pip install xarm-python-sdk")

    arm = XArmAPI(ip)
    arm.clean_warn()
    arm.clean_error()
    check_code("motion_enable(True)", arm.motion_enable(True))
    check_code("set_mode(0)", arm.set_mode(0))
    check_code("set_state(0)", arm.set_state(0))
    check_code("set_collision_sensitivity(3)", arm.set_collision_sensitivity(3))
    check_code("set_collision_rebound(True)", arm.set_collision_rebound(True))
    return arm


def recover_robot(arm: Any) -> None:
    print("Robot error/warning detected. Clearing and returning to position mode.")
    arm.clean_warn()
    arm.clean_error()
    check_code("motion_enable(True)", arm.motion_enable(True))
    check_code("set_mode(0)", arm.set_mode(0))
    check_code("set_state(0)", arm.set_state(0))


def move_robot_with_reconnect(arm: Any, args: argparse.Namespace, target_pose: list[float]):
    for attempt in range(1, args.move_retries + 1):
        try:
            if getattr(arm, "has_error", False) or getattr(arm, "has_warn", False):
                recover_robot(arm)

            print(
                "Moving to "
                f"x={target_pose[0]:.1f}, y={target_pose[1]:.1f}, z={target_pose[2]:.1f}, "
                f"r={target_pose[3]:.1f}, p={target_pose[4]:.1f}, y={target_pose[5]:.1f}"
            )
            code = arm.set_position(
                *target_pose,
                speed=args.speed,
                mvacc=args.acc,
                wait=True,
            )
            if code == 0:
                print("Move complete.")
                return arm, True

            print(f"Warning: set_position returned xArm code {code}")
            recover_robot(arm)
        except Exception as exc:
            print(f"Warning: robot move failed on attempt {attempt}/{args.move_retries}: {exc}")
            safe_disconnect(arm)
            time.sleep(args.reconnect_delay)
            arm = connect_arm(args.ip)

    return arm, False


def move_robot_hover_with_staging(arm: Any, args: argparse.Namespace, target_pose: list[float]):
    if not args.staged_hover_move:
        return move_robot_with_reconnect(arm, args, target_pose)

    safe_z = float(clamp(args.safe_z, args.min_hover_z, args.max_hover_z))
    stage_pose = [
        float(target_pose[0]),
        float(target_pose[1]),
        safe_z,
        float(target_pose[3]),
        float(target_pose[4]),
        float(target_pose[5]),
    ]

    print(f"Stage-1 move (safe Z): z={safe_z:.1f} mm")
    arm, moved = move_robot_with_reconnect(arm, args, stage_pose)
    if not moved:
        return arm, False

    print("Stage-2 move (hover Z)")
    return move_robot_with_reconnect(arm, args, target_pose)


def draw_detection(frame: np.ndarray, tag_id: int, robot_xyz: np.ndarray, inside: bool) -> None:
    status = "inside" if inside else "outside"
    text = (
        f"Tag {tag_id} robot x={robot_xyz[0]:.1f} "
        f"y={robot_xyz[1]:.1f} z={robot_xyz[2]:.1f} {status}"
    )
    color = (0, 255, 0) if inside else (0, 0, 255)
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def detect_object_pose(
    frame: np.ndarray,
    detector,
    K: np.ndarray,
    dist: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect_markers(detector, gray)
    if ids is None or len(ids) == 0:
        return None, None, corners, ids

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    rvecs, tvecs = estimate_marker_pose(corners, args.tag_size, K, dist)

    for i, marker_id in enumerate(ids.flatten()):
        if int(marker_id) != args.target_tag:
            continue
        if np.isnan(tvecs[i]).any():
            return None, None, corners, ids
        cv2.drawFrameAxes(frame, K, dist, rvecs[i], tvecs[i], args.tag_size * 0.5)
        return tvecs[i], rvecs[i], corners, ids

    return None, None, corners, ids


def run(args: argparse.Namespace) -> None:
    print("Loading calibration and workspace files...")
    K, dist = load_calibration(args.calib)
    workspace_tags = load_workspace_tags(args.workspace)
    robot_readings = load_robot_saved_readings(args.robot_yaml)
    tag_map = parse_tag_map(args.tag_map)

    R_cam_to_robot, t_cam_to_robot, workspace_robot_points = build_transform(
        workspace_tags,
        robot_readings,
        tag_map,
    )

    camera_points_mm = np.vstack([workspace_tags[tag_id] * 1000.0 for tag_id in tag_map])
    robot_points_mm = np.vstack([robot_readings[index] for index in tag_map.values()])
    mean_error, max_error = print_calibration_error(
        camera_points_mm,
        robot_points_mm,
        R_cam_to_robot,
        t_cam_to_robot,
    )

    write_tag_robot_mapping_yaml(
        args.mapping_yaml,
        workspace_tags,
        robot_readings,
        tag_map,
        R_cam_to_robot,
        t_cam_to_robot,
        mean_error,
        max_error,
    )
    print(f"Saved tag-robot mapping YAML: {args.mapping_yaml}")

    if mean_error > args.max_calibration_error:
        print(
            "Warning: calibration error is high. Check that --tag-map matches the exact "
            "order of saved robot readings in lite6_end_effector.yaml."
        )
    if args.check_only:
        return

    detector = make_detector(args.dictionary)
    source = open_source(args)
    arm = None if args.no_robot else connect_arm(args.ip)
    xy_limits = resolve_xy_limits(args, workspace_robot_points)

    print(
        "Active XY limits mm: "
        f"x:[{xy_limits[0]:.1f},{xy_limits[1]:.1f}] "
        f"y:[{xy_limits[2]:.1f},{xy_limits[3]:.1f}]"
    )

    print(f"\nWaiting for object AprilTag {args.target_tag}. Press q to quit.")
    if args.no_robot:
        print("Dry run enabled: robot will not move.")

    cv2.namedWindow("Object Pick", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Object Pick", 800, 600)

    last_move_time = 0.0
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                print("Camera frame read failed.")
                break

            cam_tvec, _, _, _ = detect_object_pose(frame, detector, K, dist, args)

            if cam_tvec is not None:
                robot_xyz = camera_to_robot(cam_tvec, R_cam_to_robot, t_cam_to_robot)
                is_inside = inside_workspace(robot_xyz, workspace_robot_points)
                draw_detection(frame, args.target_tag, robot_xyz, is_inside)
                write_object_pose_yaml(args.object_pose_yaml, args.target_tag, cam_tvec, robot_xyz, is_inside)

                print(f"Object camera xyz m: {cam_tvec}")
                print(f"Object robot xyz mm: {robot_xyz}")

                now = time.time()
                can_move = now - last_move_time >= args.move_cooldown
                if is_inside and can_move:
                    target_xyz = robot_xyz.copy()
                    target_xyz[2] = max(176.0, float(target_xyz[2]))
                    hover_z = float(target_xyz[2])

                    if not inside_xy_limits(target_xyz, xy_limits):
                        print(
                            "Target XY outside configured limits. "
                            f"x={target_xyz[0]:.1f}, y={target_xyz[1]:.1f} "
                            f"limits x:[{xy_limits[0]:.1f},{xy_limits[1]:.1f}] "
                            f"y:[{xy_limits[2]:.1f},{xy_limits[3]:.1f}]"
                        )
                        cv2.imshow("Object Pick", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        continue

                    if not inside_hover_z_limits(hover_z, args):
                        print(
                            "Target hover Z outside configured limits. "
                            f"z={hover_z:.1f} limits z:[{args.min_hover_z:.1f},{args.max_hover_z:.1f}]"
                        )
                        cv2.imshow("Object Pick", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):
                            break
                        continue

                    target_pose = [
                        float(target_xyz[0]),
                        float(target_xyz[1]),
                        float(target_xyz[2]),
                        args.roll,
                        args.pitch,
                        args.yaw,
                    ]

                    if args.no_robot:
                        print(f"Dry run target pose: {target_pose}")
                        return

                    arm, moved = move_robot_hover_with_staging(arm, args, target_pose)
                    last_move_time = time.time()
                    if moved and args.once:
                        print("Task complete.")
                        return
                elif not is_inside:
                    print("Object tag is outside the calibrated workspace. Not moving.")

            cv2.imshow("Object Pick", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        source.release()
        cv2.destroyAllWindows()
        safe_disconnect(arm)
        print("Shutdown complete.")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Move Lite6 above an object AprilTag using workspace YAML + robot YAML Kabsch calibration."
    )
    parser.add_argument("--ip", default=ROBOT_IP, help="Robot controller IP address")
    parser.add_argument(
        "--calib",
        default=str(project_root / "Camera_Caliberation" / "camera_calibration.yaml"),
        help="Camera calibration YAML",
    )
    parser.add_argument(
        "--workspace",
        default=str(project_root / "Arudo_detector" / "workspace_tags.yaml"),
        help="AprilTag workspace YAML from workspace_mapper.py",
    )
    parser.add_argument(
        "--robot-yaml",
        default=str(project_root / "Lite6_arm_pos" / "lite6_end_effector.yaml"),
        help="Lite6 saved end-effector YAML from Pos_on_april_tags.py",
    )
    parser.add_argument(
        "--tag-map",
        default="16:0,17:1,18:2,19:3",
        help="Workspace-tag to robot-saved-reading mapping, example: 16:0,17:1,8:2,19:3",
    )
    parser.add_argument(
        "--mapping-yaml",
        default=str(project_root / "Arm_to_object" / "tag_robot_mapping.yaml"),
        help="Step-4 output YAML storing 4-tag robot correspondences and camera-to-robot transform",
    )
    parser.add_argument(
        "--object-pose-yaml",
        default=str(project_root / "Arm_to_object" / "object_pose_robot.yaml"),
        help="Step-5 output YAML storing latest transformed object pose in robot frame",
    )
    parser.add_argument("--target-tag", type=int, default=TARGET_TAG_ID, help="AprilTag ID attached to object")
    parser.add_argument("--dictionary", choices=sorted(APRILTAG_DICTS), default="36h11", help="AprilTag family")
    parser.add_argument("--tag-size", type=float, default=TAG_SIZE_M, help="Physical tag side length in meters")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--realsense", action="store_true", help="Use Intel RealSense color stream")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--z-offset", type=float, default=APPROACH_Z_OFFSET_MM, help="Approach height above object in mm")
    parser.add_argument("--speed", type=float, default=MOVE_SPEED, help="Robot move speed in mm/s")
    parser.add_argument("--acc", type=float, default=MOVE_ACC, help="Robot move acceleration in mm/s^2")
    parser.add_argument(
        "--auto-xy-limits",
        action="store_true",
        default=True,
        help="Auto derive XY limits from mapped 4-tag workspace with margin",
    )
    parser.add_argument(
        "--manual-xy-limits",
        dest="auto_xy_limits",
        action="store_false",
        help="Disable auto-XY limits and use --min-x/--max-x/--min-y/--max-y",
    )
    parser.add_argument("--xy-margin", type=float, default=40.0, help="Margin in mm added around mapped workspace for auto XY limits")
    parser.add_argument("--min-x", type=float, default=-350.0, help="Minimum allowed robot X in mm for manual XY limits")
    parser.add_argument("--max-x", type=float, default=500.0, help="Maximum allowed robot X in mm for manual XY limits")
    parser.add_argument("--min-y", type=float, default=-450.0, help="Minimum allowed robot Y in mm for manual XY limits")
    parser.add_argument("--max-y", type=float, default=450.0, help="Maximum allowed robot Y in mm for manual XY limits")
    parser.add_argument("--min-hover-z", type=float, default=120.0, help="Minimum allowed hover Z in mm")
    parser.add_argument("--max-hover-z", type=float, default=420.0, help="Maximum allowed hover Z in mm")
    parser.add_argument("--safe-z", type=float, default=260.0, help="Intermediate safe Z for staged hover move in mm")
    parser.add_argument(
        "--staged-hover-move",
        action="store_true",
        default=True,
        help="Move in two stages (safe Z then hover Z) for safer path",
    )
    parser.add_argument(
        "--direct-hover-move",
        dest="staged_hover_move",
        action="store_false",
        help="Move directly to hover target without safe-Z staging",
    )
    parser.add_argument("--roll", type=float, default=ROBOT_R)
    parser.add_argument("--pitch", type=float, default=ROBOT_P)
    parser.add_argument("--yaw", type=float, default=ROBOT_Y)
    parser.add_argument("--move-retries", type=int, default=2)
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    parser.add_argument("--move-cooldown", type=float, default=2.0, help="Minimum seconds between move commands")
    parser.add_argument(
        "--max-calibration-error",
        type=float,
        default=30.0,
        help="Warn when mean Kabsch calibration error is above this many mm",
    )
    parser.add_argument("--check-only", action="store_true", help="Load YAMLs, compute Kabsch error, then exit")
    parser.add_argument("--once", action="store_true", default=True, help="Exit after first successful move")
    parser.add_argument("--continuous", dest="once", action="store_false", help="Keep running after successful moves")
    parser.add_argument("--no-robot", action="store_true", help="Detect and print target pose without moving robot")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
