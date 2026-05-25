"""
src/pose/keypoint_converter.py
-------------------------------
Simple landmark class for COCO keypoints (17 points).

COCO Keypoints (17):
0=nose, 1=leye, 2=reye, 3=lear, 4=rear,
5=lsho, 6=rsho, 7=lelb, 8=relb, 9=lwri, 10=rwri,
11=lhip, 12=rhip, 13=lknee, 14=rknee, 15=lank, 16=rank
"""

import numpy as np
from typing import Optional, List


class SmoothedLandmark:
    """Simple landmark class for COCO keypoints (normalized 0-1 coordinates)."""
    
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.visibility = float(visibility)
