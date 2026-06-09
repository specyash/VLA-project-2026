import cv2 as cv
import numpy as np

class BlockAnalyzer:
    blacklower = np.array([0,0,0])
    blackupper = np.array([180,255,60])
    # a block will be marked as black if the percentage of black pixels is between 2% and 40%
    minratio = 0.02
    maxratio = 0.4
    
    def has_mark(self,img):
        if img is None or img.size==0:
            return False
        hsv= cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask= cv.inRange(hsv, BlockAnalyzer.blacklower, BlockAnalyzer.blackupper)
        total=img.shape[0]*img.shape[1]
        if total==0:
            return False
        ratio=cv.countNonZero(mask)/total
        return BlockAnalyzer.minratio < ratio < BlockAnalyzer.maxratio
