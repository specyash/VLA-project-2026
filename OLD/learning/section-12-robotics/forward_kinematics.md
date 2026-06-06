# Forward Kinematics (FK)

**Goal:** Understand how joint angles determine where the gripper is in space — and why you rarely compute FK manually in this project.

**Prerequisites:** [robotics_fundamentals.md](robotics_fundamentals.md)

**Next:** [inverse_kinematics.md](inverse_kinematics.md)

---

## 1. Intuition

**Forward kinematics (FK):**  
If I know all joint angles, **where is the gripper?**

Input:  θ1, θ2, …, θ6  
Output: (x, y, z, roll, pitch, yaw) of the tool

**Analogy:** If you know how bent each finger joint is, you can calculate where your fingertip is.

---

## 2. Link frames and Denavit–Hartenberg (concept)

Each joint has a local coordinate frame. FK multiplies **transformation matrices** along the chain:

```
T_base_to_tool = T1 · T2 · T3 · … · T6
```

You do **not** need to implement DH tables for this course project — the xArm SDK exposes `get_position()` which is effectively FK computed inside the controller.

---

## 3. Reading pose from xArm (example)

```python
code, pose = arm.get_position(is_radian=False)
# pose = [x, y, z, roll, pitch, yaw]
```

`Lite6_arm_pos/Pos_on_april_tags.py` records these poses when teaching workspace corners.

Those readings become `ROBOT_READINGS` for homography destination points.

---

## 4. Simple 2-link planar example (learning only)

Two joints in a plane, link lengths L1, L2:

```python
import numpy as np

def fk_2link(theta1_deg, theta2_deg, L1=1.0, L2=0.8):
    t1 = np.radians(theta1_deg)
    t2 = np.radians(theta2_deg)
    x = L1*np.cos(t1) + L2*np.cos(t1+t2)
    y = L1*np.sin(t1) + L2*np.sin(t1+t2)
    return x, y

print(fk_2link(30, 40))
```

Real 6-DoF arms add orientation and 3D geometry — same idea, larger matrices.

---

## 5. Why FK matters for calibration

When you touch each AprilTag corner with the tool and save `get_position()`, you record **where the robot thinks the tool was** when aligned with that physical corner.

FK + frame definitions link **joint encoders** → **reported TCP pose** → homography target points.

If tool frame is wrong (gripper not calibrated), FK pose is wrong → picks miss.

---

## 6. FK vs vision mapping

| Source | Gives |
|--------|-------|
| Vision + homography | Target (x,y) on table from camera |
| FK / recorded poses | Robot coordinates at tag corners |
| Controller | Moves joints so TCP reaches target |

Vision does not know joint angles; arm does not know pixels — **calibration connects them**.

---

## 7. How this appears in the robotics repo

- **`Pos_on_april_tags.py`:** FK output stored in YAML.
- **`ROBOT_READINGS`:** Four XY pairs used as homography destinations.
- **Not used explicitly in v5 loop** — you command Cartesian targets directly.

---

## 8. Common beginner confusion

1. **FK is not camera math** — separate from homography.
2. **Pose vs joints** — `set_position` uses pose; `set_servo_angle` uses joints.
3. **Recorded pose error** — human alignment when teaching tags affects accuracy.

---

## 9. Practice exercises

1. Call `get_position()` in a test script while manually jogging arm (if allowed).
2. Draw 2-link FK diagram for θ1=0°, θ2=90°.
3. Explain how wrong tool Z offset affects recorded corner Y values.

---

## 10. Summary

FK maps **joints → gripper pose**. You use its **results** (saved poses) for calibration more than you implement FK yourself. The xArm controller handles FK internally when you read position or command motion.

**Next:** [inverse_kinematics.md](inverse_kinematics.md)
