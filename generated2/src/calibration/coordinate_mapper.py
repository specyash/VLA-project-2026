import cv2 as cv
import numpy as np

class CoordinateMapper:

    def __init__(self, robot_readings,corner_order, tag_to_index):
        self._robot_readings= robot_readings
        self._corner_order =corner_order
        self._tag_to_index= tag_to_index
        self._homography =None

    def update(self, workspace_pixels):
        if not workspace_pixels or len(workspace_pixels)!=4 :
            self._homography=None
            return 
        srcpts = []
        dstpts = []
        for tag_id in self._corner_order:
            px, py =workspace_pixels[tag_id]
            srcpts.append([px,py])
            idx = self._tag_to_index[tag_id]
            rx, ry = self._robot_readings[idx]
            dstpts.append([rx,ry])
        src = np.array(srcpts, dtype=np.float32)
        dst = np.array(dstpts, dtype=np.float32)
        self._homography, _ = cv.findHomography(src, dst)
    
    def cvt2robot(self, px,py):
        if self._homography is None:
            return None
        point = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv.perspectiveTransform(point, self._homography)
        robot_x, robot_y = transformed[0][0]
        return float(robot_x), float(robot_y)
    
    def is_valid(self):
        return self._homography is not None
