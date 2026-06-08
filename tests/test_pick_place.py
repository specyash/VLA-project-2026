"""Tests for pick-and-place planning helpers."""

from src.planning.pick_place import PickPlaceStep,_validate_xy,create_pick_place_plan,execute_pick_place_plan
from src.robot.simulated_robot import SimulatedRobot
from src.config import HOVER_Z, PICK_Z, DROP_Z
import pytest


def test_pick_place_step_stores_move_data() -> None:
    """A move step should store pose, speed, and description."""

    step = PickPlaceStep(
        action="move",
        x=100.0,
        y=200.0,
        z=300.0,
        speed=60.0,
        description="move above object",
    )

    assert step.action == "move"
    assert step.x == 100.0
    assert step.y == 200.0
    assert step.z == 300.0
    assert step.speed == 60.0
    assert step.description == "move above object"


def test_pick_place_step_allows_gripper_action() -> None:
    """A gripper step should not need coordinates."""

    step = PickPlaceStep(action="close_gripper", description="grasp object")

    assert step.action == "close_gripper"
    assert step.x is None
    assert step.y is None
    assert step.z is None

def test_validate_xy_accepts_two_numbers() -> None:
    """Valid coordinates should be returned as floats."""

    assert _validate_xy("object_xy", (10, 20)) == (10.0, 20.0)


def test_validate_xy_rejects_wrong_length() -> None:
    """A coordinate must have exactly x and y."""

    with pytest.raises(ValueError, match="two floats"):
        _validate_xy("object_xy", (10, 20, 30))


def test_validate_xy_rejects_non_finite_values() -> None:
    """NaN and infinity should not become robot coordinates."""

    with pytest.raises(ValueError, match="finite"):
        _validate_xy("object_xy", (10, float("inf")))

def test_create_pick_place_plan_has_expected_action_order() -> None:
    """The planner should produce the standard safe pick-and-place sequence."""

    plan = create_pick_place_plan(object_xy=(100, 200), bin_xy=(300, 400))

    assert [step.action for step in plan] == [
        "move",
        "move",
        "close_gripper",
        "move",
        "move",
        "move",
        "open_gripper",
        "move",
        "move",
    ]


def test_create_pick_place_plan_uses_object_and_bin_coordinates() -> None:
    """Object moves should use object coordinates and bin moves should use bin coordinates."""

    plan = create_pick_place_plan(object_xy=(100, 200), bin_xy=(300, 400))

    assert plan[0].x == 100.0
    assert plan[0].y == 200.0
    assert plan[0].z == HOVER_Z

    assert plan[1].x == 100.0
    assert plan[1].y == 200.0
    assert plan[1].z == PICK_Z

    assert plan[4].x == 300.0
    assert plan[4].y == 400.0
    assert plan[4].z == HOVER_Z

    assert plan[5].x == 300.0
    assert plan[5].y == 400.0
    assert plan[5].z == DROP_Z


def test_execute_pick_place_plan_runs_on_simulated_robot() -> None:
    """A full plan should produce the expected fake robot command log."""

    robot = SimulatedRobot()
    plan = create_pick_place_plan(object_xy=(100, 200), bin_xy=(300, 400))

    execute_pick_place_plan(robot, plan)

    command_log = robot.get_command_log()

    assert len(command_log) == 9
    assert command_log[0].startswith("MOVE x=100.0 y=200.0")
    assert command_log[2] == "CLOSE_GRIPPER"
    assert command_log[6] == "OPEN_GRIPPER"
    assert command_log[-1].startswith("MOVE x=0.0 y=-250.0")
