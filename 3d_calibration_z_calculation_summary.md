# Summary: 3D Kabsch Calibration & Precision Z-Height Calculation Pipeline

This document summarizes the technical design, mathematical concepts, and configuration parameters implemented in the workspace to transition from flat 2D homography to full 3D rigid-body spatial mapping. These additions ensure perfect Z-height calculations for picking and collision-free stacking.

---

## 1. Overview of the 3D Mapping Pipeline

```
  [AprilTag Detection] + [RealSense Depth (d)]
                      │
                      ▼
        [Project to 3D Camera Frame]
            xc = (u - cx) * d / fx
            yc = (v - cy) * d / fy
            zc = d
                      │
                      ▼
       [3D Kabsch Transform Mapping]
         P_robot = R * P_camera + t
                      │
                      ▼
    ┌─────────────────┴─────────────────┐
    ▼                                   ▼
 [Pick Z Calculation]                [Drop Z Stacking Calculation]
 - Read object depth                 - Stable bin Z (SURFACE_DEPTH)
 - Calculate measured block height   - Stack offset tracking (bin_offsets)
 - Apply Z range safety check        - Drop Z Offset (20.0 mm clearance)
                                     - Dynamic transit cap (safe_hover_z)
```

---

## 2. Phase 1: 3D Kabsch Calibration Setup

Historically, the pipeline used 2D homography which assumed a perfectly flat, parallel camera feed. It could not compensate for camera tilt, camera height changes, or structural depth. We replaced this with a **3D Kabsch Rigid Transformation**.

### The Calibration Process (`calibrate_kabsch.py`)
1. **Physical Reference Input**: The user enters the physical $Z$ coordinate of the table surface (relative to the robot base frame, e.g., `220.0` mm) into the script.
2. **AprilTag 3D Coordinates**: The script detects the 4 workspace AprilTags. For each tag:
   * It extracts the 2D pixel center $(u, v)$.
   * It queries the camera depth sensor to get the distance $d$ in millimeters.
   * It projects the points into 3D Camera coordinates:
     $$X_c = \frac{(u - c_x) \cdot d}{f_x}, \quad Y_c = \frac{(v - c_y) \cdot d}{f_y}, \quad Z_c = d$$
3. **Rigid Body Transformation**: Using the 3D camera points and the corresponding physical robot coordinate readings at the table surface height (where $Z_{\text{robot}} = \text{table\_z}$), the Kabsch algorithm estimates:
   * A $3\times3$ Rotation Matrix ($R$)
   * A $3\times1$ Translation Vector ($t$)
4. **Serialization**: The calibration script automatically writes these parameters, along with camera intrinsics and the table surface depth, to [kabsch_calibration.yaml](file:///d:/VLA/kabsch_calibration.yaml).

> [!NOTE]
> **Dynamic Loading**: On startup, [src/config.py](file:///d:/VLA/src/config.py) checks for `kabsch_calibration.yaml` and dynamically overrides the fallback values at runtime.

---

## 3. Phase 2: Runtime Coordinate Projection (`coordinate_mapper.py`)

At runtime, the pipeline projects 2D pixels and depth into the robot's base coordinate frame using the `KabschCoordinateMapper` class:

```python
class KabschCoordinateMapper:
    def cvt2robot_3d(self, px, py, depth_mm):
        # 1. Project to Camera Frame (in mm)
        xc = ((px - self.cx) * depth_mm) / self.fx
        yc = ((py - self.cy) * depth_mm) / self.fy
        zc = depth_mm
        camera_point = np.array([xc, yc, zc], dtype=np.float64)

        # 2. Map to Robot Frame using Rotation & Translation
        robot_point = self.R @ camera_point + self.t
        return float(robot_point[0]), float(robot_point[1]), float(robot_point[2])
```

---

## 4. Phase 3: Pick Z & Object Height Calculations

When a block is identified, the robot must grasp it at its exact center and height:

1. **Object Z-Coordinate**:
   * Center pixel $(u_o, v_o)$ and raw object depth ($d_o$) are queried.
   * Projecting yields $Z_{\text{object}}$.
2. **Block Height Calculation**:
   * The physical table surface height is defined as `PICK_Z` (calibrated, e.g. `220.0` mm).
   * The height of the block is dynamically computed:
     $$\text{measured\_height} = Z_{\text{object}} - \text{PICK\_Z}$$
3. **Safety Fallbacks**:
   * If $\text{measured\_height}$ falls within a valid block range ($10.0\text{ mm} \le h \le 80.0\text{ mm}$), the measured value is used.
   * Otherwise, it defaults to the standard fallback block height ($25.0\text{ mm}$).
   * If $Z_{\text{object}}$ is outside a safe range ($-100.0\text{ mm}$ to $500.0\text{ mm}$), it defaults to $\text{PICK\_Z} + \text{PICK\_Z\_OFFSET}$ ($300.0\text{ mm}$).

---

## 5. Phase 4: Stacking and Drop Z Calculations

To stack blocks on top of each other safely without hitting the stack horizontally or pressing down too hard, several parameters work together:

### A. Stable Bin Z-Floor
Because empty plastic bins can cause depth sensor reflections and noise (returning shallow depths like `900` mm instead of `1028` mm), the pipeline does not query live bin depth. Instead, since bins are flat on the table, we project their horizontal $(X, Y)$ base positions using the static calibrated **`SURFACE_DEPTH`** constant. This keeps the bin base height stable at exactly **`PICK_Z`** (e.g. `220.0` mm).

### B. Stacking Offset Accumulation (`bin_offsets`)
A memory dictionary `bin_offsets` tracks the cumulative height of blocks already placed in each bin:
* **Empty Bin**: $\text{offset} = 0.0\text{ mm}$
* **Drop Z formula**:
  $$\text{drop\_z} = \text{bin\_floor\_z} + \text{block\_height} + \text{current\_offset} + \text{DROP\_Z\_OFFSET}$$
* **DROP_Z_OFFSET**: Configured at **`20.0 mm`** in [src/config.py](file:///d:/VLA/src/config.py#L33) to drop the cube from a safe, elevated release position.
* **Offset increment**: After execution, the bin's offset is updated:
  $$\text{offset}_{\text{new}} = \text{offset}_{\text{old}} + \text{block\_height}$$

### C. Collision-Free Transit Height (`safe_hover_z`)
To prevent the robot arm from colliding horizontally with the growing stack of blocks, the robot transit height is computed dynamically:
$$Z_{\text{transit}} = \min(310.0, \max(\text{HOVER\_Z}, \text{actual\_drop\_z} + 5.0))$$

* **Clearance**: Ensures the bottom of the carried block travels at least **`5.0 mm`** above the top of the stack.
* **Reach Safety Cap**: Caps the height at **`310.0 mm`** to prevent exceeding the xArm Lite6's vertical joints/kinematic limit, avoiding `ControllerError, code: 21` halts.

---

## Summary of Key Variables in `src/config.py`

| Variable | Current Value | Description |
| :--- | :--- | :--- |
| `HOVER_Z` | `276.0 mm` | Default vertical transit/hover Z height. |
| `PICK_Z` | `220.0 mm` | Calibrated Z coordinate of the table surface (dynamically loaded). |
| `DROP_Z` | `220.0 mm` | Calibrated drop base Z coordinate (dynamically loaded). |
| `PICK_Z_OFFSET` | `80.0 mm` | Safety Z buffer added to `PICK_Z` for empty-grip fallbacks. |
| `DROP_Z_OFFSET` | `20.0 mm` | Vertical clearance above the stack top where the gripper releases the block. |
| `safe_hover_z` | Dynamic (Max `310.0`) | Calculated travel height above the bins to prevent stack collision. |
