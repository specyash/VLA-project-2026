'''Real xArm robot adapter implementation.'''

from __future__ import annotations
import time

from src.config import ROBOT_IP, MOVE_ACC, GRIPPER_SDK_WAIT_SEC, HOME_POSE, RECOVER_WAIT_SEC, RECOVER_MOVE_SPEED
from src.voice.voice_worker import VoiceWorker

class XArmRobotAdapter:

    def __init__(self, robot_ip:str | None = None, gripper_ip:str | None = None, dry_run:bool = False, use_voice:bool = True) -> None:
        self.robot_ip = robot_ip or ROBOT_IP
        self.gripper_ip = gripper_ip
        self.dry_run = dry_run
        self.arm = None
        self._command_log: list[str] = []
        self._emergency_stopped = False

        self.voice = None
        if use_voice:
            self.voice = VoiceWorker()
            self.voice.start()

        if self.dry_run:
            self._record("DRY RUN MODE")
            return

        self._connect()

    def move_to_pose(self, x:float, y:float, z:float, r:float = -180.0, p:float = 0.0, yaw:float = -90.0, speed:float = 20.0, motion_type:str = "linear", description: str | None = None) -> None:
        command = (f"MOVE x={x:.1f} y={y:.1f} z={z:.1f} r={r:.1f} p={p:.1f} yaw={yaw:.1f} speed={speed:.1f} motion_type={motion_type}")

        self._ensure_ready_for_motion()

        if self.dry_run or self.arm is None:
            self._record(command, description)
            return

        if self.arm.error_code != 0:
            raise RuntimeError(f"Robot error: {self.arm.error_code}")

        if motion_type == "joint":
            # Compute inverse kinematics for joint angles (in degrees by default)
            ik_code, joints = self.arm.get_inverse_kinematics([x, y, z, r, p, yaw], input_is_radian=False)
            if ik_code == 0:
                print(f"[xArm] Joint motion planned. Target joints: {[round(j, 2) for j in joints]}")
                code = self.arm.set_servo_angle(angle=joints, speed=speed, wait=True, is_radian=False)
            else:
                print(f"[xArm Warning] IK failed with code {ik_code} for target x={x:.1f} y={y:.1f} z={z:.1f}. Falling back to Cartesian linear move.")
                code = self.arm.set_position(x=x, y=y, z=z, roll=r, pitch=p, yaw=yaw, speed=speed, mvacc=MOVE_ACC, wait=True)
        else:
            code = self.arm.set_position(x=x, y=y, z=z, roll=r, pitch=p, yaw=yaw, speed=speed, mvacc=MOVE_ACC, wait=True)

        self._record(command, description)

        if code != 0 or self.arm.error_code != 0:
            raise RuntimeError(f"Robot arm failed with code: {code}, error code: {self.arm.error_code}")

    def home(self) -> None:
        self._ensure_ready_for_motion()
        home_x, home_y, home_z, home_r, home_p, home_yaw = HOME_POSE
        self.move_to_pose(x=home_x, y=home_y, z=home_z, r=home_r, p=home_p, yaw=home_yaw, motion_type="joint", description="Returning to home position.")

    def open_gripper(self, description: str | None = None) -> None:
        self._gripper_action(state=0, command_name="OPEN_GRIPPER", description=description)

    def close_gripper(self, description: str | None = None) -> None:
        self._gripper_action(state=1, command_name="CLOSE_GRIPPER", description=description)

    def recover(self) -> None:
        self._record("RECOVER")

        if self.dry_run or self.arm is None:
            self.open_gripper()
            self._record("HOME")
            self._emergency_stopped = False
            return

        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(RECOVER_WAIT_SEC)

        self._safe_open_gripper()
        self._emergency_stopped = False

        home_x, home_y, home_z, home_r, home_p, home_yaw = HOME_POSE
        self.move_to_pose(
            x=home_x,
            y=home_y,
            z=home_z,
            r=home_r,
            p=home_p,
            yaw=home_yaw,
            speed=RECOVER_MOVE_SPEED,
            motion_type="joint",
            description="Returning to home position after recovery."
        )

    def emergency_stop(self, reason: str = "manual") -> None:
        """Halt the arm immediately and block further motion. Recover() must be called for the arm to start working again."""

        self._record(f"EMERGENCY_STOP reason={reason}")
        self._emergency_stopped = True

        if self.dry_run or self.arm is None:
            self._record("OPEN_GRIPPER")
            return

        if hasattr(self.arm, "emergency_stop"):
            self.arm.emergency_stop()
        else:
            self.arm.set_state(4)

        self.arm.motion_enable(False)
        self._safe_open_gripper()

    def is_emergency_stopped(self) -> bool:
        """Return True if motion is blocked until recover()."""

        return self._emergency_stopped

    def get_command_log(self) -> list[str]:
        return list(self._command_log)

    def disconnect(self) -> None:
        if not self.dry_run and self.arm is not None:
            self.arm.disconnect()
            self.arm = None
            self._record("DISCONNECTED")

    def _connect(self) -> None:
        try:
            from xarm.wrapper import XArmAPI
        except ImportError as error:
            raise RuntimeError(f"xArm API not installed: {error}") from error
        
        self.arm = XArmAPI(self.robot_ip, do_not_open=False)
        self.arm.clean_warn()
        self.arm.clean_error()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self._record(f"CONNECTED: {self.robot_ip}")

    def _gripper_action(self, state:int, command_name:str, description: str | None = None) -> None:
        
        '''Implement gripper action.
            state 0 for open
            state 1 for close
            Sends signal to TO 0 pin (TGPIO digital pin 0)'''

        if self.dry_run or self.arm is None:
            self._record(command_name, description)
            return

        code = self.arm.set_tgpio_digital(0, state)
        if code != 0:
            raise RuntimeError(f"Gripper SDK command {command_name} failed with code: {code}")

        self._record(command_name, description)
        time.sleep(GRIPPER_SDK_WAIT_SEC)

    def _ensure_ready_for_motion(self) -> None:

        if self._emergency_stopped:
            raise RuntimeError("Robot is emergency-stopped. Call recover() before moving.")

    def _safe_open_gripper(self) -> None:
        """Open the gripper without raising during emergency handling."""

        try:
            self.open_gripper()
        except Exception as error:
            self._record(f"OPEN_GRIPPER_FAILED {error}")

    def _record(self, message:str, description: str | None = None) -> None:
        '''Record command to log.'''
        print(message)
        self._command_log.append(message)
        if self.voice:
            spoken_text = description if description else self._format_speech(message)
            self.voice.speak(spoken_text)

    def _format_speech(self, message:str) -> str:
        if message.startswith("MOVE "):
            parts = message.split()
            coords = {}
            for part in parts[1:]:
                if "=" in part:
                    key, val = part.split("=", 1)
                    try:
                        coords[key] = float(val)
                    except ValueError:
                        coords[key] = val
            x = coords.get("x", 0.0)
            y = coords.get("y", 0.0)
            z = coords.get("z", 0.0)
            return f"Moving to X {x:.0f}, Y {y:.0f}, Z {z:.0f}."
        elif message == "HOME":
            return "Moving to home position."
        elif message == "OPEN_GRIPPER":
            return "Opening gripper."
        elif message == "CLOSE_GRIPPER":
            return "Closing gripper."
        elif message == "RECOVER":
            return "Recovering robot."
        elif message.startswith("CONNECTED:"):
            return "Robot connected."
        elif message == "DISCONNECTED":
            return "Robot disconnected."
        elif message.startswith("EMERGENCY_STOP"):
            return "Emergency stop."
        return message