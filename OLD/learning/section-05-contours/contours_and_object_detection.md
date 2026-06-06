# Contours and Object Localization

## Intuition
A contour is the boundary line around connected foreground pixels.

## Basic contour flow
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    area = cv2.contourArea(c)
    if area < 500: continue
    x, y, w, h = cv2.boundingRect(c)
    cx, cy = x + w//2, y + h//2
```

## Why robots care
Contours provide:
- approximate object location
- approximate size
- centroid for grasp targeting

## Shape hints
- polygon approximation (`cv2.approxPolyDP`)
- circularity from area/perimeter

## Repo connection
Main pipeline uses YOLO for object detection, but contour logic is still useful for quick fallback detectors and mask cleanup.

## Beginner confusion
- Threshold quality controls contour quality.
- Tiny specks produce many false contours.

## Practice
Detect largest contour in a binary image and draw its bounding rectangle.
