"""Base robot adapter interface placeholder."""

from __future__ import annotations
from typing import Protocol

class RobotAdapter(Protocol):

    def home(self) -> None:
        """move the robot to the home position"""

    def move_to_pose(self, x:float, y:float, z:float, r:float= -180.0, p:float=0.0, yaw:float= -90.0, speed:float=10.0, motion_type:str = "linear", description: str | None = None) -> None:
        """move the robot to a specific pose"""

    def open_gripper(self) -> None:
        """open the gripper"""

    def close_gripper(self) -> None:
        """close the gripper"""

    def recover(self) -> None:
        """recover the robot from an error state"""

    def emergency_stop(self, reason: str = "manual") -> None:
        """Halt motion immediately during collision or unsafe behavior."""

    def is_emergency_stopped(self) -> bool:
        """Return True if the robot is halted and waiting for recovery."""

    def get_command_log(self) -> list[str]:
        """get the command log"""