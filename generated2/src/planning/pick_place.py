"""Pick-and-place planner placeholder."""

from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from src.config import HOME_POSE, HOVER_Z, PICK_Z, DROP_Z, ROBOT_R, ROBOT_P, ROBOT_YAW, DEFAULT_MOVE_SPEED, SLOW_MOVE_SPEED

@dataclass
class PickPlaceStep:

    #Action name eg. HOME, MOVE, OPEN_GRIPPER, CLOSE_GRIPPER, RECOVER
    action:str

    # Optional pose values. Gripper actions do not need these.
    x: float | None = None
    y: float | None = None
    z: float | None = None

    # Tool orientation.
    r: float | None = None
    p: float | None = None
    yaw: float | None = None

    # Movement speed for move actions.
    speed: float | None = None

    # Short human-readable reason for this step.
    description: str = ""

def _validate_xy(name:str, xy:tuple[float, float]) -> tuple[float, float]:
    if(len(xy) != 2):
        raise ValueError(f"{name} must be a tuple of two floats")

    x, y = xy

    if not isfinite(x) or not isfinite(y):
        raise ValueError(f"{name} must be a finite number")

    return x, y

def _validate_xyz(name:str, xyz:tuple[float, float, float]) -> tuple[float, float, float]:
    if(len(xyz) != 3):
        raise ValueError(f"{name} must be a tuple of three floats")

    x, y, z = xyz

    if not isfinite(x) or not isfinite(y) or not isfinite(z):
        raise ValueError(f"{name} must contain finite numbers")

    return x, y, z

def create_pick_place_plan(
    object_xyz: tuple[float, float, float],
    bin_xy: tuple[float, float],
    drop_z: float | None = None
) -> list[PickPlaceStep]:

#Create a sequence of steps to move object to bin in a safe way

    object_x, object_y, object_z = _validate_xyz("object_xyz", object_xyz)
    bin_x, bin_y = _validate_xy("bin_xy", bin_xy)
    home_x, home_y, home_z, home_r, home_p, home_yaw = HOME_POSE
    actual_drop_z = drop_z if drop_z is not None else DROP_Z

    return [
        PickPlaceStep(
            action="move",
            x=object_x,
            y=object_y,
            z=HOVER_Z,
            r=ROBOT_R,
            p=ROBOT_P,
            yaw=ROBOT_YAW,
            speed=DEFAULT_MOVE_SPEED,
            description="move above object",
        ),
        PickPlaceStep(
            action="move",
            x=object_x,
            y=object_y,
            z=object_z,
            r=ROBOT_R,
            p=ROBOT_P,
            yaw=ROBOT_YAW,
            speed=SLOW_MOVE_SPEED,
            description="move down to pick",
        ),
        PickPlaceStep(action="close_gripper", description="grasp object"),
        PickPlaceStep(
            action="move",
            x=object_x,
            y=object_y,
            z=HOVER_Z,
            r=ROBOT_R,
            p=ROBOT_P,
            yaw=ROBOT_YAW,
            speed=DEFAULT_MOVE_SPEED,
            description="lift object",
        ),
        PickPlaceStep(
            action="move",
            x=bin_x,
            y=bin_y,
            z=HOVER_Z,
            r=ROBOT_R,
            p=ROBOT_P,
            yaw=ROBOT_YAW,
            speed=DEFAULT_MOVE_SPEED,
            description="move above bin",
        ),
        PickPlaceStep(
            action="move",
            x=bin_x,
            y=bin_y,
            z=actual_drop_z,
            r=ROBOT_R,
            p=ROBOT_P,
            yaw=ROBOT_YAW,
            speed=SLOW_MOVE_SPEED,
            description="move down to drop",
        ),
        PickPlaceStep(action="open_gripper", description="release object"),
        PickPlaceStep(
            action="move",
            x=bin_x,
            y=bin_y,
            z=HOVER_Z,
            r=ROBOT_R,
            p=ROBOT_P,
            yaw=ROBOT_YAW,
            speed=DEFAULT_MOVE_SPEED,
            description="move above bin after release",
        ),
        PickPlaceStep(
            action="move",
            x=home_x,
            y=home_y,
            z=home_z,
            r=home_r,
            p=home_p,
            yaw=home_yaw,
            speed=DEFAULT_MOVE_SPEED,
            description="return home",
        ),
    ]

def execute_pick_place_plan(robot, plan:list[PickPlaceStep]) -> None:

    for step in plan:
        execute_step(robot,step)

def execute_step(robot, step:PickPlaceStep) -> None:

    if step.action == "move":
        if step.x is None or step.y is None or step.z is None:
            raise ValueError("Move step must have x, y, and z coordinates")

        robot.move_to_pose(x=step.x, y=step.y, z=step.z,
        r=step.r if step.r is not None else ROBOT_R,
        p=step.p if step.p is not None else ROBOT_P, 
        yaw=step.yaw if step.yaw is not None else ROBOT_YAW, 
        speed=step.speed if step.speed is not None else DEFAULT_MOVE_SPEED)

        return

    if step.action == "close_gripper":
        robot.close_gripper()
        return

    if step.action == "open_gripper":
        robot.open_gripper()
        return

    if step.action == "home":
        robot.home()
        return

    raise ValueError(f"Unknown action: {step.action}")