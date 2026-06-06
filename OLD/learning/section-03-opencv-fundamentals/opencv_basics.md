# OpenCV Basics

Goal: learn core `cv2` operations used in the repo.

## Intuition
OpenCV is your robot's "image toolbox": read frame, draw info, show results, repeat.

## Core functions
```python
import cv2
img = cv2.imread('sample.png')
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Video loop:
```python
cap = cv2.VideoCapture(0)
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow('cam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
```

## Drawing overlays
```python
cv2.rectangle(frame, (100,100), (200,200), (0,255,0), 2)
cv2.circle(frame, (150,150), 4, (0,0,255), -1)
cv2.putText(frame, 'target', (100,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
```

Why overlays? Real-time debugging and operator trust.

## How this appears in this repo
- Main window and HUD use `cv2.imshow` continuously.
- Bounding boxes and labels are drawn in `draw_annotations`.
- Key controls (`q`, `1`, `2`, `3`, `c`) use `cv2.waitKey`.

## Beginner confusion
- `imshow` requires `waitKey` to refresh.
- BGR order, not RGB.

## Practice
1. Read a webcam frame and draw FPS text.
2. Add a red circle at the frame center.
