"""
src/visualization/coco_skeleton_drawer.py
------------------------------------------
Draw COCO skeleton with rainbow-colored landmarks.
"""

import cv2
import numpy as np
from typing import List


# COCO skeleton edges (17 keypoints)
COCO_EDGES = [
    (5, 7), (7, 9),      # left arm
    (6, 8), (8, 10),     # right arm
    (5, 6),              # shoulders
    (5, 11), (6, 12),    # torso
    (11, 12),            # hips
    (11, 13), (13, 15),  # left leg
    (12, 14), (14, 16)   # right leg
]

# Rainbow colors for 17 COCO keypoints (BGR format)
RAINBOW_COLORS = [
    (0, 0, 255),        # 0: nose - RED
    (0, 127, 255),      # 1: leye - ORANGE
    (0, 255, 255),      # 2: reye - YELLOW
    (0, 255, 127),      # 3: lear - YELLOW-GREEN
    (0, 255, 0),        # 4: rear - GREEN
    (127, 255, 0),      # 5: lsho - GREEN-CYAN
    (255, 255, 0),      # 6: rsho - CYAN
    (255, 127, 0),      # 7: lelb - CYAN-BLUE
    (255, 0, 0),        # 8: relb - BLUE
    (255, 0, 127),      # 9: lwri - BLUE-INDIGO
    (255, 0, 255),      # 10: rwri - INDIGO
    (127, 0, 255),      # 11: lhip - INDIGO-VIOLET
    (0, 0, 127),        # 12: rhip - VIOLET
    (64, 0, 255),       # 13: lknee - VIOLET-RED
    (0, 64, 255),       # 14: rknee - RED-ORANGE
    (0, 128, 255),      # 15: lank - ORANGE
    (0, 192, 255),      # 16: rank - ORANGE-YELLOW
]


def draw_coco_skeleton(img: np.ndarray, 
                       kpts: np.ndarray,
                       radius: int = 4,
                       thickness: int = 2) -> np.ndarray:
    """Draw COCO skeleton with rainbow-colored edges and landmarks."""
    out = img.copy()
    kpts = np.array(kpts, dtype=float)
    
    # Only use first 17 COCO keypoints (ignore any extra keypoints)
    if len(kpts) > 17:
        kpts = kpts[:17]

    # Draw edges with rainbow colors based on the keypoints they connect
    for (i, j) in COCO_EDGES:
        if i < len(kpts) and j < len(kpts):
            pt1 = tuple(np.int32(kpts[i]))
            pt2 = tuple(np.int32(kpts[j]))
            # Use color from the first keypoint of the edge
            color = RAINBOW_COLORS[i % len(RAINBOW_COLORS)]
            cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw landmarks with rainbow colors
    inner_radius = max(1, radius - 2)
    for i, p in enumerate(kpts):
        pt = tuple(np.int32(p))
        color = RAINBOW_COLORS[i % len(RAINBOW_COLORS)]
        # Draw filled circle with rainbow color
        cv2.circle(out, pt, radius, color, -1)
        # Draw white outline
        cv2.circle(out, pt, radius, (255, 255, 255), 1)

    return out


def put_multiline_text(img: np.ndarray,
                       lines: List[str],
                       org: tuple,
                       line_h: int = 22,
                       font=cv2.FONT_HERSHEY_SIMPLEX,
                       scale: float = 0.6,
                       color: tuple = (255, 255, 255),
                       thickness: int = 1) -> np.ndarray:
    """
    Put multiple lines of text on image.
    
    Parameters
    ----------
    img : np.ndarray
        Image to draw on
    lines : list
        List of text strings
    org : tuple
        (x, y) origin position
    line_h : int
        Line height in pixels
    font : int
        OpenCV font
    scale : float
        Font scale
    color : tuple
        BGR color
    thickness : int
        Text thickness
    
    Returns
    -------
    np.ndarray
        Image with text drawn
    """
    out = img.copy()
    x, y = org
    for i, t in enumerate(lines):
        cv2.putText(out, t, (x, y + i * line_h), font, scale, color, thickness, cv2.LINE_AA)
    return out
