# OpenCV Image Processing

Goal: understand preprocessing before detection.

## Intuition
Preprocessing is "cleaning the camera signal" so downstream algorithms are more stable.

## Common operations
```python
blur = cv2.GaussianBlur(frame, (5,5), 0)
median = cv2.medianBlur(frame, 5)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 60, 150)
_, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
```

- Gaussian blur: smooth noise.
- Median blur: strong against salt-and-pepper noise.
- Canny: edge extraction.
- Threshold: separate foreground/background.

## Morphology quick intro
```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
```

## Robotics use
- Cleaner masks -> better contour centers.
- Better edges -> better marker boundaries.

## Repo mapping
- `is_marked_with_black()` uses HSV thresholding and ratio checks.
- Depth HUD uses `cv2.Canny` on quantized depth layers.

## Common confusion
- Over-blur can erase small objects.
- Fixed thresholds fail under changing light.

## Practice
Try three Gaussian kernel sizes (3,5,9) and compare edge quality.
