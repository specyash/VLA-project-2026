# Coordinate Systems

## Intuition
A coordinate frame is a ruler + orientation attached to an observer.

Robotics always has multiple observers:
- image frame (pixels)
- camera frame (meters)
- robot base frame (millimeters)

## Why this exists
The camera sees pixels; the robot moves in physical units. We must transform between frames.

## 2D and 3D
- 2D point: (x, y)
- 3D point: (x, y, z)

## Example
```python
pixel = (640, 360)
# Later converted with homography -> robot_xy_mm
```

## Repo mapping
`pixel_to_robot()` transforms image point to robot XY using a homography built from AprilTag correspondences.

## Beginner confusion
Same point has different numbers in different frames.

## Practice
Write three labels for a table corner: pixel, camera xyz, robot xyz.
