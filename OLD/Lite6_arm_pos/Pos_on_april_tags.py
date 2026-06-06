#!/usr/bin/env python3
"""
Read the xArm Lite6 end-effector pose with respect to the robot base.

The xArm API returns TCP/end-effector pose as:
    x, y, z, roll, pitch, yaw

Position is in millimeters. Angles are in degrees by default.

Examples:
    python3 Pos_on_april_tags.py --ip 192.168.1.152
    python3 Pos_on_april_tags.py --ip 192.168.1.152 --rate 5
    python3 Pos_on_april_tags.py --ip 192.168.1.152 --output lite6_end_effector.yaml
    python3 Pos_on_april_tags.py --ip 192.168.1.152 --once
"""

from __future__ import annotations

import argparse
import select
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_ROBOT_IP = "192.168.1.152"
DEFAULT_OUTPUT = "lite6_end_effector.yaml"


def check_code(name: str, code: int) -> None:
    if code != 0:
        print(f"Warning: {name} returned xArm code {code}")


def safe_disconnect(arm: Any) -> None:
    try:
        arm.disconnect()
    except Exception as exc:
        print(f"Warning: disconnect failed: {exc}")


def connect_arm(ip: str):
    try:
        from xarm.wrapper import XArmAPI
    except ImportError:
        sys.exit("xArm Python SDK not found. Install it with: pip install xarm-python-sdk")

    arm = XArmAPI(ip)
    arm.clean_warn()
    arm.clean_error()

    # Mode 2 is xArm manual / joint teaching mode.
    check_code("motion_enable(True)", arm.motion_enable(True))
    check_code("set_mode(2)", arm.set_mode(2))
    check_code("set_state(0)", arm.set_state(0))
    return arm


def reconnect_arm(arm: Any | None, ip: str, delay: float):
    if arm is not None:
        safe_disconnect(arm)

    print(f"Reconnecting to xArm Lite6 in {delay:.1f}s...")
    time.sleep(delay)
    arm = connect_arm(ip)
    print("Reconnected.")
    return arm


def read_end_effector_pose(arm, is_radian: bool = False) -> list[float]:
    code, pose = arm.get_position(is_radian=is_radian)
    if code != 0:
        raise RuntimeError(f"Failed to read end-effector pose. xArm error code: {code}")
    return pose


def read_pose_with_reconnect(
    arm: Any,
    args: argparse.Namespace,
    consecutive_failures: int,
) -> tuple[Any, list[float] | None, int]:
    try:
        pose = read_end_effector_pose(arm, is_radian=args.radian)
        if consecutive_failures:
            print("Pose read recovered.")
        return arm, pose, 0
    except Exception as exc:
        consecutive_failures += 1
        print(
            f"Warning: pose read failed ({consecutive_failures}/{args.max_failures}): {exc}"
        )

        if consecutive_failures < args.max_failures:
            time.sleep(args.retry_delay)
            return arm, None, consecutive_failures

        try:
            arm = reconnect_arm(arm, args.ip, args.reconnect_delay)
            return arm, None, 0
        except Exception as reconnect_exc:
            print(f"Warning: reconnect failed: {reconnect_exc}")
            time.sleep(args.reconnect_delay)
            return arm, None, consecutive_failures


def print_pose(pose: list[float], is_radian: bool = False) -> None:
    x, y, z, roll, pitch, yaw = pose
    angle_unit = "rad" if is_radian else "deg"
    print(
        "End-effector pose wrt base | "
        f"x={x:+9.3f} mm  y={y:+9.3f} mm  z={z:+9.3f} mm  "
        f"roll={roll:+8.3f} {angle_unit}  "
        f"pitch={pitch:+8.3f} {angle_unit}  "
        f"yaw={yaw:+8.3f} {angle_unit}"
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5


def write_pose_yaml(path: str, samples: list[list[float]], is_radian: bool = False) -> None:
    if not samples:
        return

    names = ["x", "y", "z", "roll", "pitch", "yaw"]
    angle_unit = "rad" if is_radian else "deg"
    columns = list(zip(*samples))
    avgs = {name: mean(list(values)) for name, values in zip(names, columns)}
    stds = {name: std(list(values)) for name, values in zip(names, columns)}

    lines = [
        "lite6_end_effector:",
        f"  created_unix_time: {time.time():.3f}",
        "  frame: base",
        "  position_units: mm",
        f"  angle_units: {angle_unit}",
        "  pose:",
        f"    samples: {len(samples)}",
        f"    x: {avgs['x']:.8f}",
        f"    y: {avgs['y']:.8f}",
        f"    z: {avgs['z']:.8f}",
        f"    roll: {avgs['roll']:.8f}",
        f"    pitch: {avgs['pitch']:.8f}",
        f"    yaw: {avgs['yaw']:.8f}",
        f"    std_x: {stds['x']:.8f}",
        f"    std_y: {stds['y']:.8f}",
        f"    std_z: {stds['z']:.8f}",
        f"    std_roll: {stds['roll']:.8f}",
        f"    std_pitch: {stds['pitch']:.8f}",
        f"    std_yaw: {stds['yaw']:.8f}",
        "  saved_readings:",
    ]

    for i, pose in enumerate(samples):
        x, y, z, roll, pitch, yaw = pose
        lines.extend(
            [
                f"    {i}:",
                f"      x: {x:.8f}",
                f"      y: {y:.8f}",
                f"      z: {z:.8f}",
                f"      roll: {roll:.8f}",
                f"      pitch: {pitch:.8f}",
                f"      yaw: {yaw:.8f}",
            ]
        )

    Path(path).write_text("\n".join(lines) + "\n")


def enter_pressed() -> bool:
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    sys.stdin.readline()
    return True


def run(args: argparse.Namespace) -> None:
    print(f"Connecting to xArm Lite6 at {args.ip}...")
    arm = connect_arm(args.ip)
    print("Connected. Robot set to manual mode.")
    print("Reading end-effector pose relative to base.")
    print(f"Press Enter to save the latest reading to {args.output}.")

    saved_samples: list[list[float]] = []
    last_pose = None
    consecutive_failures = 0
    try:
        if args.once:
            pose = None
            for attempt in range(args.max_failures):
                arm, pose, consecutive_failures = read_pose_with_reconnect(
                    arm, args, consecutive_failures
                )
                if pose is not None:
                    break
                print(f"Retrying one-shot read ({attempt + 1}/{args.max_failures})...")
            if pose is None:
                raise RuntimeError("Could not read pose after retries.")
            print_pose(pose, is_radian=args.radian)
            write_pose_yaml(args.output, [pose], is_radian=args.radian)
            print(f"Saved 1 reading to {args.output}")
            return

        delay = 1.0 / max(args.rate, 0.1)
        print("Press Ctrl+C to stop.")
        while True:
            arm, pose, consecutive_failures = read_pose_with_reconnect(
                arm, args, consecutive_failures
            )
            if pose is not None:
                last_pose = pose
                print_pose(last_pose, is_radian=args.radian)

            if last_pose is not None and enter_pressed():
                saved_samples.append(list(last_pose))
                write_pose_yaml(args.output, saved_samples, is_radian=args.radian)
                print(f"Saved reading {len(saved_samples)} to {args.output}")

            time.sleep(delay)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if saved_samples:
            write_pose_yaml(args.output, saved_samples, is_radian=args.radian)
            print(f"Final YAML saved: {args.output} ({len(saved_samples)} readings)")
        elif last_pose is not None:
            print("No readings saved. Press Enter while running to save a pose.")
        safe_disconnect(arm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Set Lite6 to manual mode and print end-effector pose with respect to the base."
    )
    parser.add_argument("--ip", default=DEFAULT_ROBOT_IP, help="Robot controller IP address")
    parser.add_argument("--rate", type=float, default=2.0, help="Loop print rate in Hz")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output YAML file")
    parser.add_argument("--once", action="store_true", help="Print one pose and exit")
    parser.add_argument("--radian", action="store_true", help="Return orientation in radians instead of degrees")
    parser.add_argument(
        "--max-failures",
        type=int,
        default=3,
        help="Consecutive pose-read failures before reconnecting",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=0.25,
        help="Seconds to wait after a failed pose read before retrying",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before reconnecting after repeated failures",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
