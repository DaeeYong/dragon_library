import numpy as np
import cv2
def get_roi(img : np.ndarray, x0 : int, y0 : int, w : int, h : int) -> np.ndarray:
    if x0 < 0 or y0 < 0:
        return False
    
    if w > 0 and h > 0:
        roi = img[y0: y0 + h, x0:x0 + w]
        return roi
    
    return False