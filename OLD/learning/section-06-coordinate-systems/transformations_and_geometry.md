# Transformations and Geometry

## Intuition
A transform answers: "Where is this point in another frame?"

Two basic operations:
- translation (shift)
- rotation (turn)

In homogeneous coordinates:
```python
import numpy as np
T = np.array([[1,0,tx],
              [0,1,ty],
              [0,0,1 ]], dtype=float)
p = np.array([x,y,1.0])
p2 = T @ p
```

## Robotics significance
Every pick command uses geometry:
camera/object estimate -> robot target pose.

## Perspective transforms
Planar mapping uses 3x3 homography matrix H.

## Repo mapping
`cv2.findHomography` computes H from 4 workspace tags.
`cv2.perspectiveTransform` applies H to object center and bin points.

## Common confusion
- Rotation order matters in 3D.
- Units must be consistent (pixels vs mm).

## Practice
Given two points and a translation (10, -5), compute transformed coordinates manually.
