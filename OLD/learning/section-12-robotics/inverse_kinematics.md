# Inverse Kinematics (IK)

**Goal:** Understand how desired gripper poses become joint motions — and why this project uses Cartesian commands instead of solving IK in Python.

**Prerequisites:** [forward_kinematics.md](forward_kinematics.md)

**Next:** [robot_motion_planning.md](robot_motion_planning.md)

---

## 1. Intuition

**Inverse kinematics (IK):**  
Given desired gripper pose (x, y, z, roll, pitch, yaw), **what joint angles achieve it?**

FK: joints → pose  
IK: pose → joints

**Analogy:** “Put my hand on this cup” — your brain solves IK, not you consciously.

---

## 2. Why IK is harder than FK

- **Multiple solutions** — same pose, different elbow-up/down configurations.
- **No solution** — target unreachable or orientation impossible.
- **Singularities** — infinite joint rates for tiny Cartesian motion.

Industrial arms hide this in the **controller**.

---

## 3. How this repo commands motion

You call:

```python
arm.set_position(x, y, z, r, p, yaw, speed=60, wait=True)
```

The xArm firmware:
1. Receives Cartesian target.
2. Solves IK internally (or Jacobian-based control).
3. Plans joint trajectory.
4. Executes while checking limits and errors.

**You do not call an IK solver in `automated_pipeline_v5.py`.**

---

## 4. When you would implement IK in Python

- Custom arms without Cartesian API.
- Research on redundant arms.
- ROS MoveIt pipelines.

This course favors **reliable integration** over implementing IK from scratch.

---

## 5. Vision → IK data flow (conceptual)

```
(cx, cy) pixel
    → homography → (rx, ry) mm
    → choose z, r, p, yaw
    → set_position(rx, ry, z, ROBOT_R, ROBOT_P, ROBOT_Y)
           │
           ▼
    [controller IK + motion]
           │
           ▼
    joint motion → physical pick
```

---

## 6. Error codes and IK failures

```python
if arm.set_position(...) != 0 or arm.error_code != 0:
    return False  # move_robot in v5
```

Failures may mean:
- Out of reach.
- Singularity near workspace boundary.
- Collision / self-collision check (if enabled).
- E-stop or not enabled.

`recover_robot()` clears errors and returns home.

---

## 7. Orientation fixed in v5

Constant `ROBOT_R, ROBOT_P, ROBOT_Y` reduces IK difficulty — controller always approaches with same wrist angle.

Changing orientation per object would require checking IK feasibility for each grasp.

---

## 8. How this appears in the robotics repo

| Location | IK role |
|----------|---------|
| `move_robot()` | Wrapper around `set_position` |
| `execute_pick_and_place()` | Sequence of Cartesian targets |
| `object_pick.py` | More complex pose targets + limits |
| MoveIt (not in repo) | Alternative IK in ROS labs |

---

## 9. Common beginner confusion

1. **Homography is not IK** — homography gives XY; arm adds Z and orientation.
2. **wait=True** — Python blocks until motion done; IK already solved during move.
3. **Dry run** — `arm=None` skips real IK entirely.
4. **Millimeters** — wrong units → IK target invalid.

---

## 10. Practice exercises

1. Command a point clearly inside workspace vs far away — compare error codes.
2. Read xArm docs for `set_position` return values.
3. Explain why fixed yaw simplifies sorting cubes.

---

## 11. Summary

IK converts **desired TCP poses** to **joint trajectories**. In this project, the **xArm controller** performs IK; your job is to supply accurate **Cartesian targets** from vision and safe motion sequencing.

**Next:** [robot_motion_planning.md](robot_motion_planning.md)
