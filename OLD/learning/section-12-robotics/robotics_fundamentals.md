# Robotics Fundamentals

**Goal:** Build mental models for manipulator arms, end effectors, and pick-and-place without prior robotics background.

**Prerequisites:** [Section 6 — Coordinate systems](../section-06-coordinate-systems/coordinate_systems.md)

**Next:** [forward_kinematics.md](forward_kinematics.md)

**Official references:**
- [UFACTORY xArm user manual](https://www.ufactory.cc/) (hardware-specific)
- [Introduction to robot manipulators (general)](https://en.wikipedia.org/wiki/Robot_manipulator)

---

## 1. What is a robotic manipulator?

A **manipulator** is a chain of **links** connected by **joints** that positions a **tool** (gripper) in space.

```
Base → Joint1 → Link → Joint2 → … → End effector (gripper)
```

Your lab uses a **UFACTORY xArm Lite 6** — 6 revolute joints → **6 degrees of freedom (6-DoF)** for the tool pose (position + orientation).

---

## 2. Key vocabulary

| Term | Meaning |
|------|---------|
| **Joint space** | Robot described by joint angles (θ1…θ6) |
| **Cartesian / task space** | Tool pose (x, y, z, roll, pitch, yaw) |
| **End effector** | Gripper or tool at the last link |
| **Reach workspace** | Volume where the arm can physically go |
| **Singularity** | Configurations where some motions are impossible or unstable |
| **Payload** | Max mass the arm can lift |

---

## 3. Intuition: your arm vs the robot arm

Your shoulder-elbow-wrist moves the hand. The robot controller moves joints so the **gripper center** (TCP — tool center point) reaches a target pose.

Vision says: “object at (x,y) on the table.”  
Controller says: “joints, go there.”

---

## 4. Top-down pick-and-place (this project)

Course and repo intentionally use **simple** manipulation:

- Camera looks down (roughly) at the table.
- Gripper approaches from **above** (vertical-ish).
- Fixed orientation: `ROBOT_R, ROBOT_P, ROBOT_Y` constant in v5.
- No complex 6-DoF reorientation per object.

**Stages:**

```
HOME → hover above object → descend → close gripper →
lift → hover above bin → descend → open gripper → HOME
```

See `execute_pick_and_place()` in `automated_pipeline_v5.py`.

---

## 5. Why fixed orientation?

Sorting colored blocks on a flat table does not need rotating the gripper 90° if blocks are similar size and gripper is symmetric enough.

**Benefits:** fewer failures, simpler IK, faster moves.

---

## 6. Z heights in this repo

| Constant | Typical role |
|----------|----------------|
| `Z_LIMIT` (176 mm) | Near-table pick/place height |
| `HOVER_Z` (276 mm) | Safe travel height over obstacles |
| `HOME_POSE` | Rest configuration above workspace |
| `bin_offsets[bin_id]` | Stack blocks higher (+25 mm per place) |

Z is **not** from depth in v5 — it is **scripted** from table calibration experience.

---

## 7. Gripper as part of the robot system

Two control paths:

1. **HTTP** to `GRIPPER_IP` — custom peripheral.
2. **GPIO** on xArm — `set_cgpio_digital(0, state)` fallback.

Open/close timing uses fixed `sleep` — no force sensing in main pipeline.

---

## 8. ASCII: perception → action boundary

```
Vision output:  (rx, ry) in mm, object class
                      │
                      ▼
Manipulation:    (rx, ry, Z_LIMIT) + fixed R,P,Y
                      │
                      ▼
Hardware:        xArm motors + gripper
```

---

## 9. How this appears in the robotics repo

| Concept | Code |
|---------|------|
| Connect arm | `XArmAPI(ROBOT_IP)` in `main()` |
| Cartesian move | `arm.set_position(x,y,z,r,p,yaw, ...)` |
| Pick sequence | `execute_pick_and_place()` |
| Recovery | `recover_robot()` |
| Dry run | `arm is None` → moves skipped |

---

## 10. Common beginner confusion

1. **mm vs m** — xArm SDK uses **millimeters** and **degrees** in this project.
2. **Base frame** — coordinates are relative to robot base, not camera.
3. **Tool frame** — set_position usually means TCP pose, not joint angles.
4. **Hover first** — skipping hover risks collisions with other objects.

---

## 11. Practice exercises

1. Sketch side view: table, cube, hover Z, pick Z.
2. Read `HOME_POSE` and explain why Z=300 is safer than Z=176 for transit.
3. List what happens if `rx, ry` is outside arm reach.

---

## 12. Summary

Robotics here means **moving a known TCP through known waypoints** in response to vision. The arm has joints; you command **Cartesian poses**; the controller handles joint motion internally.

**Next:** [forward_kinematics.md](forward_kinematics.md)
