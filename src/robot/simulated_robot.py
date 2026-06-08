"""Simulation-only robot adapter placeholder."""


from __future__ import annotations

class SimulatedRobot:

    def __init__(self) -> None:
        self._command_log: list[str] = []

    def home(self) -> None:
        self._record("HOME")

    def move_to_pose(
        self,
        x: float,
        y: float,
        z: float,
        r: float = -180.0,
        p: float = 0.0,
        yaw: float = -90.0,
        speed: float = 60.0,
    ) -> None:
        command = (f"MOVE x={x:.1f} y={y:.1f} z={z:.1f} r={r:.1f} p={p:.1f} yaw={yaw:.1f} speed={speed:.1f}")
        self._record(command)

    def open_gripper(self) -> None:
        self._record("OPEN_GRIPPER")

    def close_gripper(self) -> None:
        self._record("CLOSE_GRIPPER")

    def recover(self) -> None:
        self._record("RECOVER")

    def get_command_log(self) -> list[str]:
        return self._command_log.copy()

    def _record(self, command: str) -> None:
        print(command)
        self._command_log.append(command)