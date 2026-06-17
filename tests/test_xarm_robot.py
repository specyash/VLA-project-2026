from src.robot.xarm_robot import XArmRobotAdapter
from src.config import HOME_POSE
import pytest


def test_xarm_dry_run_logs_move_without_hardware() -> None:
    """Dry-run mode should log moves without needing a real robot."""

    robot = XArmRobotAdapter(dry_run=True, use_voice=False)

    robot.move_to_pose(x=100.0, y=200.0, z=300.0)

    command_log = robot.get_command_log()

    assert len(command_log) == 2
    assert "DRY RUN MODE" in command_log[0]
    assert command_log[1].startswith("MOVE x=100.0 y=200.0 z=300.0")


def test_xarm_home_uses_configured_home_pose_in_dry_run() -> None:
    """Home should call move_to_pose with HOME_POSE values."""

    robot = XArmRobotAdapter(dry_run=True, use_voice=False)
    robot.home()

    command_log = robot.get_command_log()

    home_x, home_y, home_z = HOME_POSE[:3]
    assert command_log[-1].startswith(f"MOVE x={home_x:.1f} y={home_y:.1f} z={home_z:.1f}")


def test_xarm_dry_run_logs_gripper_commands() -> None:
    """Gripper commands should also work in dry-run mode."""

    robot = XArmRobotAdapter(dry_run=True, use_voice=False)

    robot.open_gripper()
    robot.close_gripper()

    assert robot.get_command_log()[-2:] == ["OPEN_GRIPPER", "CLOSE_GRIPPER"]

def test_xarm_emergency_stop_blocks_further_moves_in_dry_run() -> None:
    """After emergency stop, movement should be blocked until recover()."""

    robot = XArmRobotAdapter(dry_run=True, use_voice=False)
    robot.emergency_stop(reason="collision")

    assert robot.is_emergency_stopped() is True

    with pytest.raises(RuntimeError, match="emergency-stopped"):
        robot.move_to_pose(10, 20, 30)


def test_xarm_recover_clears_emergency_state_in_dry_run() -> None:
    """Recover should release the emergency lock."""

    robot = XArmRobotAdapter(dry_run=True, use_voice=False)
    robot.emergency_stop(reason="collision")
    robot.recover()

    assert robot.is_emergency_stopped() is False
    robot.move_to_pose(10, 20, 30)
    assert robot.get_command_log()[-1].startswith("MOVE x=10.0 y=20.0 z=30.0")