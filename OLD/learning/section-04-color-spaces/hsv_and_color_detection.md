# HSV and Color Detection

## Intuition
RGB mixes brightness and color. HSV separates them:
- H: hue (color type)
- S: saturation (color strength)
- V: value (brightness)

That makes color filtering easier under illumination changes.

## Basic example
```python
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
lower = (35, 40, 40)   # green low
upper = (85, 255, 255) # green high
mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(frame, frame, mask=mask)
```

## Noise handling
- blur before threshold
- morphology open/close
- area filtering on connected components

## Robotics sorting use
Color is a direct attribute -> bin mapping rule.

## Repo mapping
`is_marked_with_black()` converts ROI to HSV and uses `cv2.inRange` for black ratio to tag `_marked` vs `_plain` blocks.

## Beginner confusion
- Hue wraps around for red (needs two ranges).
- Thresholds must be tuned per camera and lighting.

## Practice
Create masks for red, green, blue from webcam and display each.
