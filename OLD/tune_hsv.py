import cv2
import numpy as np
import time

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
if REALSENSE_AVAILABLE:
    cap.release()
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipe.start(cfg)
    def read_frame():
        frames = pipe.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf: return False, None
        return True, np.asanyarray(cf.get_data())
else:
    def read_frame():
        return cap.read()

print("\n--- LIVE HSV COLOR TUNER ---")
print("1. Place the cubes under the camera in their normal lighting.")
print("2. Adjust the sliders until ONLY the cube you want is WHITE in the 'Mask' window.")
print("3. Press 'q' to quit and get your exact color ranges.\n")

cv2.namedWindow('HSV Tuner', cv2.WINDOW_NORMAL)
cv2.resizeWindow('HSV Tuner', 600, 300)
cv2.moveWindow('HSV Tuner', 0, 0)
cv2.imshow('HSV Tuner', np.zeros((1, 600, 3), np.uint8)) 

cv2.namedWindow('Result')
cv2.moveWindow('Result', 650, 0)

cv2.namedWindow('Original Feed')
cv2.moveWindow('Original Feed', 0, 350)

cv2.namedWindow('Mask (White is what robot sees)')
cv2.moveWindow('Mask (White is what robot sees)', 650, 350)

cv2.createTrackbar('HMin', 'HSV Tuner', 0, 179, nothing)
cv2.createTrackbar('SMin', 'HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('VMin', 'HSV Tuner', 0, 255, nothing)
cv2.createTrackbar('HMax', 'HSV Tuner', 179, 179, nothing)
cv2.createTrackbar('SMax', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('VMax', 'HSV Tuner', 255, 255, nothing)

cv2.waitKey(100) # Give Windows OS a moment to register the trackbars

while True:
    ok, frame = read_frame()
    if not ok or frame is None:
        continue
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    try:
        h_min = cv2.getTrackbarPos('HMin', 'HSV Tuner')
        s_min = cv2.getTrackbarPos('SMin', 'HSV Tuner')
        v_min = cv2.getTrackbarPos('VMin', 'HSV Tuner')
        h_max = cv2.getTrackbarPos('HMax', 'HSV Tuner')
        s_max = cv2.getTrackbarPos('SMax', 'HSV Tuner')
        v_max = cv2.getTrackbarPos('VMax', 'HSV Tuner')
    except cv2.error:
        # Window was closed or not ready
        break
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Resize displays so they all fit on screen together
    disp_frame = cv2.resize(frame, (640, 360))
    disp_mask = cv2.resize(mask, (640, 360))
    disp_result = cv2.resize(result, (640, 360))
    
    cv2.imshow('Original Feed', disp_frame)
    cv2.imshow('Mask (White is what robot sees)', disp_mask)
    cv2.imshow('Result', disp_result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\n[SUCCESS] Copy these values into visualize_and_pick.py COLOR_RANGES:")
        print(f"(({h_min}, {s_min}, {v_min}), ({h_max}, {s_max}, {v_max}))\n")
        break

if REALSENSE_AVAILABLE:
    pipe.stop()
else:
    cap.release()
cv2.destroyAllWindows()
