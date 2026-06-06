# OpenCV Video Pipeline

Goal: build a robust frame-by-frame real-time loop.

## Canonical loop
```python
while True:
    ok, frame = read_frame()
    if not ok: continue
    # preprocess
    # detect
    # annotate
    cv2.imshow('Main View', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
```

## Why frame loops exist
Robots operate in continuous time. Each frame updates world state.

## Pipeline pattern
```
Capture -> Process -> Decide -> Render -> Handle key
```

## Performance tips
- Keep heavy inference in worker thread.
- Avoid copying full frames unnecessarily.
- Keep UI drawing lightweight.

## Repo mapping
`automated_pipeline_v5.py` loop does:
1. frame capture
2. tag update
3. YOLO push/results
4. state-machine decision
5. robot action and overlays

## Beginner confusion
- Blocking robot calls (`wait=True`) pause the loop.
- `cv2.VideoCapture` fallback has no depth frame.

## Practice
Implement a loop that shows grayscale and edge windows side-by-side.
