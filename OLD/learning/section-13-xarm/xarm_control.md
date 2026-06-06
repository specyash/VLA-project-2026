# xArm Robot Control (XArmAPI)

**Goal:** Understand how this project connects to, enables, and commands the UFACTORY xArm.

**Prerequisites:** [robot_motion_planning.md](../section-12-robotics/robot_motion_planning.md)

**Next:** [Section 14 — Multithreaded robotics](../section-14-multithreading/multithreaded_robotics.md)

**Official references:**
- [xArm Python SDK (UFACTORY)](https://github.com/xArm-Developer/xArm-Python-SDK)
- [xArm API documentation](https://www.ufactory.cc/)

---

## 1. Intuition: SDK as a remote control API

The arm has an **embedded controller**. Your Python script sends **high-level commands** over Ethernet:

```
Laptop (automated_pipeline_v5.py)
        │  TCP/IP 192.168.1.152
        ▼
xArm control box → motors → gripper IO / external HTTP gripper
```

You are not driving PWM pins directly — you talk to the **API**.

---

## 2. Connection sequence (repo)

```python
from xarm.wrapper import XArmAPI

arm = XArmAPI(ROBOT_IP, do_not_open=False)  # "192.168.1.152"
arm.clean_warn()
arm.clean_error()
arm.motion_enable(True)
arm.set_mode(0)    # position control mode
arm.set_state(0)   # ready state
gripper_action(arm, 0)  # open
arm.set_position(*HOME_POSE, wait=True)
```

| Step | Purpose |
|------|---------|
| `clean_warn` / `clean_error` | Clear fault flags from previous session |
| `motion_enable(True)` | Power motors |
| `set_mode(0)` | Position control (not teach mode) |
| `set_state(0)` | Ready to accept commands |
| `HOME_POSE` | Known safe configuration |

If connection fails → `arm = None` → **dry-run** (vision only).

---

## 3. Cartesian moves

```python
def move_robot(arm, x, y, z, r, p, yw, speed=60):
    if arm:
        if arm.error_code != 0:
            return False
        code = arm.set_position(x, y, z, r, p, yw, speed=speed, wait=True)
        if code != 0 or arm.error_code != 0:
            return False
    return True
```

**Units:** mm and degrees (default in this project).

**`wait=True`:** blocks until motion finished or fault.

**Pose constants in v5:**

```python
ROBOT_R, ROBOT_P, ROBOT_Y = -178.0, 0.0, -88.0
HOME_POSE = [0.0, -250.0, 300.0, -180.0, 0.0, -88.0]
```

---

## 4. Gripper control (dual path)

### Path A — HTTP gripper (preferred in v5)

```python
url = f"http://{GRIPPER_IP}/gripper/{state}"  # 0=open, 1=close
requests.get(url, timeout=2)
time.sleep(3)
```

Separate device on `192.168.0.147` — common in lab setups with custom end effector.

### Path B — xArm control GPIO

```python
arm.set_cgpio_digital(0, state)
time.sleep(1)
```

Used if HTTP fails.

**Gap:** no feedback that grasp succeeded — only time delays.

---

## 5. Error handling and recovery

```python
def recover_robot(arm):
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_state(0)
    gripper_action(arm, 0)
    arm.set_position(*HOME_POSE, speed=60, wait=True)
```

Call path: failed `move_robot` during pick → `recover_robot` → returns `False` to caller.

**Safety practices (lab):**
- Keep e-stop accessible.
- Clear workspace before homing.
- Verify `HOME_POSE` clears table and bins.

---

## 6. Teaching poses for calibration

`Lite6_arm_pos/Pos_on_april_tags.py`:

- Puts arm in **teach mode** (`set_mode(2)`) optionally.
- Reads `get_position()` when operator aligns tool with tag.
- Saves YAML for later `ROBOT_READINGS`.

This links **physical corners** to **numbers** homography needs.

---

## 7. `test_connect.py`

Minimal connectivity check — run before full pipeline to verify IP and SDK install:

```python
from xarm.wrapper import XArmAPI
arm = XArmAPI("192.168.1.152")
# read state / position
```

---

## 8. ASCII: xArm in full stack

```
Vision (rx, ry, z plan)
        │
        ▼
 move_robot / set_position
        │
        ▼
 xArm controller (IK, limits)
        │
        ▼
 Motors + gripper
```

---

## 9. Common beginner confusion

1. **Wrong IP / subnet** — laptop must reach `192.168.1.x` lab network.
2. **Not enabled** — forgetting `motion_enable` → no movement.
3. **Mode 2 teach vs 0 position** — wrong mode rejects commands.
4. **Dry-run success** — `arm=None` still returns True from `move_robot` — misleading for tests.
5. **Collision** — Cartesian line through table if hover skipped.

---

## 10. Practice exercises

1. Connect and print `arm.get_position()` at home.
2. Jog Z only +10 mm with `set_position` — confirm direction matches expectation.
3. Trace HTTP vs GPIO path when Wi-Fi to gripper is unplugged.

---

## 11. Summary

xArm control in this repo is **connect → enable → repeated `set_position`** with fixed orientation and scripted Z. Gripper is **HTTP-first**. Recovery returns arm to **HOME** after faults.

**Next:** [multithreaded_robotics.md](../section-14-multithreading/multithreaded_robotics.md)
