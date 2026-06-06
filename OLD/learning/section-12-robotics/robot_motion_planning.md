# Robot Motion Planning

**Goal:** Understand waypoints, trajectories, and the pick-place state machine used in this repository.

**Prerequisites:** [inverse_kinematics.md](inverse_kinematics.md)

**Next:** [Section 13 — xArm control](../section-13-xarm/xarm_control.md)

---

## 1. Intuition: motion planning vs “go there”

**Motion planning** chooses **how** to move from A to B safely:
- Avoid table collision.
- Stay within joint limits.
- Smooth speed profiles.

**This repo** uses a **waypoint script** — a fixed list of poses — not a full planner like ROS MoveIt with obstacle maps.

That matches course guidance: *simple, reliable manipulation scores higher than fragile complexity.*

---

## 2. Waypoint sequence in `execute_pick_and_place`

```python
steps = [
    (rx, ry, HOVER_Z, 60, None),                              # approach object high
    (rx, ry, safe_z, 40, lambda: gripper_action(arm, 1)),     # descend, close
    (rx, ry, HOVER_Z, 60, lambda: voice.speak(...)),          # lift
    (bx, by, HOVER_Z, 60, None),                              # travel to bin
    (bx, by, drop_z, 40, lambda: gripper_action(arm, 0)),   # descend, open
    (bx, by, HOVER_Z, 60, None),                              # lift from bin
    (*HOME_POSE[:3], 60, lambda: voice.speak(...)),           # home XY/Z
]
```

Each step: `(x, y, z, speed, optional_callback)`.

**Callbacks** run **after** move completes — e.g. close gripper only when descended.

---

## 3. ASCII: height profile (side view)

```
Z
^     HOVER_Z ────────────────╮           ╭────────────
│              ╭──────────────╯           ╰──╮
│              │ pick Z                      │ drop Z (+ offset)
│              ╰─ object                      ╰─ bin
└────────────────────────────────────────────────────► XY
      home area          object XY              bin XY
```

---

## 4. Stacking logic (`bin_offsets`)

```python
current_offset = bin_offsets.get(bin_id, 0.0)
drop_z = min(safe_z + current_offset, HOVER_Z - 10.0)
# after place:
bin_offsets[bin_id] = current_offset + 25.0
```

Each successful place raises next drop height by 25 mm for that bin — reduces stacking collisions.

Reset when user presses `r` (calibration reset).

---

## 5. Speed parameter

`speed=60` vs `40` — slower on precision segments (descend/grasp).

Tune if:
- Objects slip (slower descend).
- Cycle time too long (faster hover moves only).

---

## 6. Recovery as planning fallback

```python
if not move_robot(...):
    return recover_robot(arm)
```

`recover_robot`: clear errors, open gripper, go `HOME_POSE`.

**Missing vs course rubric:** no **retry same object** — only abort/recover.

---

## 7. Application state machine (task level)

Separate from motion waypoints — **what task is running**:

| `app_state` | Behavior |
|-------------|----------|
| `MENU` | Wait for 1/2/3 |
| `AUTO_SORT` | Loop picks until empty ~20 frames |
| `CUSTOM_PICK_COLOR` | User selects object |
| `CUSTOM_PICK_BIN` | User selects bin |
| `CHAT` | VLM queries |

Motion planning is **nested inside** `AUTO_SORT` when a target exists.

---

## 8. Blocking behavior

`wait=True` on each `set_position` means the **main perception loop stops** during entire pick (~seconds to tens of seconds).

Implications:
- No mid-pick vision updates.
- Voice/YOLO threads still run, but state machine does not re-enter AUTO_SORT logic until pick returns.

---

## 9. How this appears in the robotics repo

| Function | Role |
|----------|------|
| `move_robot` | Single Cartesian move + error check |
| `gripper_action` | Open/close with HTTP/GPIO |
| `execute_pick_and_place` | Full sort cycle |
| `recover_robot` | Fault handler |
| `run_pipeline.py` | Simpler linear pick (one object, one bin) |

---

## 10. Common beginner confusion

1. **Lambda in steps** — runs after move; if move fails, callback may not run.
2. **drop_z cap** — `HOVER_Z - 10` limits stack height growth.
3. **HOME only XYZ** — last step uses first three HOME components; orientation from `ROBOT_R,P,Y` in `move_robot`.
4. **No path around obstacles** — straight-line Cartesian segments assumed clear.

---

## 11. Practice exercises

1. Time one full `execute_pick_and_place` cycle.
2. Draw state diagram: MENU → AUTO_SORT → pick → MENU.
3. List failure modes with no retry (object not grasped).

---

## 12. Summary

Motion planning here is **scripted waypoints** at safe heights plus **per-bin Z offsets**. Task-level **state machine** decides *what* to pick; `execute_pick_and_place` decides *how* to move.

**Next:** [xarm_control.md](../section-13-xarm/xarm_control.md)
