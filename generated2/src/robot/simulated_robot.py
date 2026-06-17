"""Simulation-only robot adapter placeholder."""


from __future__ import annotations

class SimulatedRobot:

    def __init__(self) -> None:
        self._command_log: list[str] = []
        self._emergency_stopped = False

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
        self._record("OPEN_GRIPPER")
        self._record("HOME")
        self._emergency_stopped = False

    def emergency_stop(self, reason: str = "manual") -> None:
        """Record an immediate halt command."""

        self._record(f"EMERGENCY_STOP reason={reason}")
        self._record("OPEN_GRIPPER")
        self._emergency_stopped = True

    def is_emergency_stopped(self) -> bool:
        """Return True if motion is blocked until recover()."""

        return self._emergency_stopped

    def _ensure_ready_for_motion(self) -> None:
        """Block movement after an emergency stop until recovery."""

        if self._emergency_stopped:
            raise RuntimeError("Robot is emergency-stopped. Call recover() before moving.")


    def get_command_log(self) -> list[str]:
        return self._command_log.copy()

    def _record(self, command: str) -> None:
        print(command)
        self._command_log.append(command)