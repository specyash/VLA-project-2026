# Homography and Workspace Mapping

## Intuition
If your table is flat, you can map camera pixels on that plane directly to robot-table coordinates using one matrix.

Think of it like flattening a tilted photo of a sheet of paper.

## Core math
Homography is a 3x3 matrix H such that:
`p_table ~ H * p_image`

## OpenCV example
```python
import cv2, numpy as np
src = np.array([[100,100],[500,100],[500,400],[100,400]], dtype=np.float32)
dst = np.array([[0,0],[300,0],[300,200],[0,200]], dtype=np.float32)
H, _ = cv2.findHomography(src, dst)
pt = np.array([[[320,250]]], dtype=np.float32)
out = cv2.perspectiveTransform(pt, H)
```

## Why it exists in robotics
It turns an object center `(cx,cy)` into workspace `(x_mm,y_mm)` for pick-and-place.

## Bird's-eye view
`cv2.warpPerspective` can visualize a top-down view of workspace.

## Repo mapping
`PipelineState._compute_homography()` uses tag pixels (16,17,18,19) -> robot corner readings.

## Limitations
- assumes a planar surface
- invalid if camera or table moves significantly

## Practice
Compute H from four corner points, then transform a clicked pixel.
