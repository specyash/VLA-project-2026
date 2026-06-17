"""Tests for the simulated robot adapter."""

from src.robot.simulated_robot import SimulatedRobot


def test_simulated_robot_records_commands() -> None:
    """The fake robot should store every command in order."""

    robot = SimulatedRobot()

    robot.home()
    robot.move_to_pose(x=100.0, y=200.0, z=300.0)
    robot.close_gripper()
    robot.open_gripper()
    robot.recover()

    assert robot.get_command_log() == [
        "HOME",
        "MOVE x=100.0 y=200.0 z=300.0 r=-180.0 p=0.0 yaw=-90.0 speed=60.0",
        "CLOSE_GRIPPER",
        "OPEN_GRIPPER",
        "RECOVER",
        "OPEN_GRIPPER",
        "HOME",
    ]


def test_command_log_returns_copy() -> None:
    """External code should not mutate the robot's internal command log."""

    robot = SimulatedRobot()
    robot.home()

    command_log = robot.get_command_log()
    command_log.append("FAKE_COMMAND")

    assert robot.get_command_log() == ["HOME"]