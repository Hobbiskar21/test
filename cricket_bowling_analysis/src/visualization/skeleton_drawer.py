"""
src/visualization/skeleton_drawer.py
--------------------------------------
Draws the skeleton on each frame using COCO format (17 keypoints).

Uses COCO skeleton edges and visualization style from scripts/draw_utils.py.
Color convention — all values are BGR (OpenCV order):
    White   (255, 255, 255)  — skeleton lines
    Black   (  0,   0,   0)  — joint fill
    White   (255, 255, 255)  — joint outline
"""

from __future__ import annotations

import cv2
import numpy as np
from src.visualization.coco_skeleton_drawer import draw_coco_skeleton, put_multiline_text
from src.visualization.feature_priority_display import (
    draw_feature_priority_box, draw_view_status_bar, draw_sync_info
)


# COCO landmark indices (17 keypoints)
NOSE         = 0
L_SHOULDER   = 5
R_SHOULDER   = 6
L_ELBOW      = 7
R_ELBOW      = 8
L_WRIST      = 9
R_WRIST      = 10
L_HIP        = 11
R_HIP        = 12
L_KNEE       = 13
R_KNEE       = 14
L_ANKLE      = 15
R_ANKLE      = 16

# HUD layout constants
_HUD_LINE_HEIGHT  = 22
_HUD_FONT         = cv2.FONT_HERSHEY_SIMPLEX
_HUD_FONT_SCALE   = 0.55
_HUD_FONT_THICKNESS = 1





def draw_skeleton(
    frame: np.ndarray,
    landmarks,
    width: int,
    height: int,
    min_visibility: float = 0.5,
    line_thickness: int = 2,
    circle_radius: int = 4,
    angle_annotations: dict[str, float] | None = None,
    phase_label: str | None = None,
    bowling_style: str | None = None,
    velocities: dict | None = None,
    angle_metadata: dict | None = None,
    view_info: dict | None = None,
    sync_info: dict | None = None,
    frame_number: int | None = None,
) -> np.ndarray:
    """
    Draw the skeleton on a single frame using COCO format with feature priority display.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame (not modified — a copy is returned).
    landmarks : list of COCO landmarks (17 points)
        Per-frame landmark list (17 COCO landmarks).
    width, height : int
        Frame dimensions in pixels.
    min_visibility : float
        Landmarks below this visibility score are skipped (0–1).
    line_thickness : int
        Stroke width for skeleton lines.
    circle_radius : int
        Radius of joint-dot circles.
    angle_annotations : dict[str, float], optional
        Joint-angle values to display in the HUD.
    phase_label : str, optional
        Current phase name to show in the HUD.
    bowling_style : str, optional
        Bowling style classification to show in the HUD.
    velocities : dict, optional
        Velocity metrics to display.
    angle_metadata : dict, optional
        Metadata for angles (view, confidence, etc).
    view_info : dict, optional
        View information (view_used, view_fallback).
    sync_info : dict, optional
        Multi-view sync information.
    frame_number : int, optional
        Frame number to display at top left.

    Returns
    -------
    np.ndarray
        Annotated BGR frame (copy — original is not modified).
    """
    out = frame.copy()
    
    # Draw frame number at top left in big text
    if frame_number is not None:
        cv2.putText(out, f"Frame: {frame_number}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    if landmarks is None:
        return out

    # Accept direct COCO pixel coordinates so the render path can match the scripts exactly.
    if isinstance(landmarks, np.ndarray):
        out = draw_coco_skeleton(out, landmarks, radius=circle_radius, thickness=line_thickness)
        
        # Draw feature priority box at BOTTOM if metadata available
        if angle_metadata:
            h = out.shape[0]
            box_position = (10, h - 250)  # Bottom of screen
            out = draw_feature_priority_box(out, angle_annotations or {}, angle_metadata, position=box_position)
            
            # Draw phase label ABOVE the box in BLACK
            if phase_label:
                phase_y = h - 270
                cv2.putText(out, f"PHASE: {phase_label.upper()}", (10, phase_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        
        return out
    
    # Convert COCO landmarks to numpy array (17, 2) with visibility check
    coco_kpts = np.zeros((17, 2), dtype=np.float32)
    valid_count = 0
    for i in range(min(17, len(landmarks))):
        lm = landmarks[i]
        if lm is not None and lm.visibility > min_visibility:
            # Only set if visibility is good
            # Landmarks are in normalized coordinates (0-1), scale to pixel coordinates
            coco_kpts[i] = [lm.x * width, lm.y * height]
            valid_count += 1
        else:
            # Set to 0,0 for invalid keypoints (will be filtered in draw_coco_skeleton)
            coco_kpts[i] = [0, 0]
    
    # Debug: print if we have valid keypoints
    if valid_count == 0:
        print(f"[SKELETON] WARNING: No valid keypoints found (all visibility < {min_visibility})")
        return out
    
    # Draw COCO skeleton
    out = draw_coco_skeleton(out, coco_kpts, radius=circle_radius, thickness=line_thickness)
    
    # Draw feature priority box at BOTTOM if metadata available
    if angle_metadata:
        h = out.shape[0]
        box_position = (10, h - 250)  # Bottom of screen
        out = draw_feature_priority_box(out, angle_annotations or {}, angle_metadata, position=box_position)
        
        # Draw phase label ABOVE the box in BLACK
        if phase_label:
            phase_y = h - 270
            cv2.putText(out, f"PHASE: {phase_label.upper()}", (10, phase_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    return out
