"""
src/biomechanics/bowling_style_detector.py
--------------------------------------------
Validates camera angle and classifies bowling style (FRONT_ON, SIDE_ON, MIXED, UNKNOWN)
using multi-signal analysis of body alignment.
"""

import warnings
import numpy as np
from numpy.typing import ArrayLike
import math

# COCO landmark indices (17 keypoints)
# 0=nose, 5=lsho, 6=rsho, 7=lelb, 8=relb, 9=lwri, 10=rwri,
# 11=lhip, 12=rhip, 13=lknee, 14=rknee, 15=lank, 16=rank
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12
L_ANKLE = 15
R_ANKLE = 16
L_FOOT_INDEX = 15  # Use ankle as reference (COCO doesn't have foot_index)
R_FOOT_INDEX = 16  # Use ankle as reference
L_HEEL = 15  # Use ankle as reference
R_HEEL = 16  # Use ankle as reference


def validate_camera_angle(landmarks_sequence, frame_width: float, frame_height: float) -> tuple[bool, str]:
    """
    Detects camera angle and returns whether it's valid for analysis.
    Now supports side-on videos with limited features.

    Uses the horizontal distance between shoulders across the first 30 frames.
    If median shoulder width (normalized by frame width) is below 0.10, it's side-on.

    Parameters
    ----------
    landmarks_sequence : list of landmark lists
        Per-frame landmarks from YOLO COCO Pose (17 keypoints).
    frame_width : float
        Frame width in pixels.
    frame_height : float
        Frame height in pixels.

    Returns
    -------
    tuple[bool, str]
        (is_valid, message)
    """
    if frame_width <= 0:
        return True, "Invalid frame width - proceeding with side-on analysis"

    shoulder_widths = []

    # Check first 30 frames
    for i, lm in enumerate(landmarks_sequence[:30]):
        if lm is None:
            continue

        # Get left and right shoulders
        if i < len(lm) and L_SHOULDER < len(lm) and R_SHOULDER < len(lm):
            l_shoulder = lm[L_SHOULDER]
            r_shoulder = lm[R_SHOULDER]

            if l_shoulder.visibility >= 0.5 and r_shoulder.visibility >= 0.5:
                # Horizontal distance between shoulders
                shoulder_width = abs(r_shoulder.x - l_shoulder.x) * frame_width
                shoulder_widths.append(shoulder_width)

    if not shoulder_widths:
        return True, "Could not detect shoulders - proceeding with available features"

    # Use median for stability
    median_shoulder_width = np.median(shoulder_widths)
    normalized_width = median_shoulder_width / frame_width

    # Always return True - support both angles but track camera type
    if normalized_width < 0.10:
        return True, "Side-on camera angle detected - showing available features"

    return True, "Camera angle valid"


def get_camera_angle_type(landmarks_sequence, frame_width: float, frame_height: float) -> str:
    """
    Detects camera angle type: 'front_back' or 'side_on'.

    Returns
    -------
    str
        Either 'front_back' for front/back facing or 'side_on' for side-on angle
    """
    if frame_width <= 0:
        return 'side_on'

    shoulder_widths = []

    # Check first 30 frames
    for i, lm in enumerate(landmarks_sequence[:30]):
        if lm is None:
            continue

        if i < len(lm) and L_SHOULDER < len(lm) and R_SHOULDER < len(lm):
            l_shoulder = lm[L_SHOULDER]
            r_shoulder = lm[R_SHOULDER]

            if l_shoulder.visibility >= 0.5 and r_shoulder.visibility >= 0.5:
                shoulder_width = abs(r_shoulder.x - l_shoulder.x) * frame_width
                shoulder_widths.append(shoulder_width)

    if not shoulder_widths:
        return 'side_on'

    median_shoulder_width = np.median(shoulder_widths)
    normalized_width = median_shoulder_width / frame_width

    return 'side_on' if normalized_width < 0.10 else 'front_back'


def _compute_angle_from_points(p1: tuple, p2: tuple) -> float:
    """
    Compute the angle of a line from p1 to p2 relative to horizontal.
    Returns angle in degrees (0-90 range for absolute value).
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(abs(dy), abs(dx))
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def _get_landmark_pos(lm_list, idx: int, frame_width: float, frame_height: float) -> tuple | None:
    """
    Extract (x, y) pixel position from a landmark if visible.
    Returns None if landmark is not visible or out of bounds.
    """
    if lm_list is None or idx >= len(lm_list):
        return None
    
    lm = lm_list[idx]
    if lm.visibility < 0.5:
        return None
    
    return (lm.x * frame_width, lm.y * frame_height)


def classify_bowling_style(landmarks_at_landing, frame_width: float, frame_height: float) -> tuple[str, dict]:
    """
    Classifies bowling style as FRONT_ON, SIDE_ON, MIXED, or UNKNOWN
    using 5 independent signals with weighted voting.
    
    MIXED action detection:
    - Lower body (hips/legs) faces 3 o'clock (90° angle)
    - Upper body (shoulders) faces 1 o'clock (30-45° angle)
    - This creates the characteristic "chest-on" mixed action
    
    Parameters
    ----------
    landmarks_at_landing : landmark list
        Landmarks at the front foot landing frame.
    frame_width : float
        Frame width in pixels.
    frame_height : float
        Frame height in pixels.
    
    Returns
    -------
    tuple[str, dict]
        (style, breakdown_dict)
        breakdown_dict contains:
            - 'style': classification result
            - 'front_score': total front-on score
            - 'side_score': total side-on score
            - 'mixed_score': total mixed action score
            - 'signals': dict of individual signal values
    """
    if landmarks_at_landing is None:
        return "UNKNOWN", {
            'style': 'UNKNOWN',
            'front_score': 0.0,
            'side_score': 0.0,
            'mixed_score': 0.0,
            'signals': {}
        }
    
    signals = {}
    signal_scores = {}  # (front_weight, side_weight, mixed_weight) for each signal
    
    # Signal 1: Front foot angle (weight 0.20)
    front_foot_ratio = None
    r_heel = _get_landmark_pos(landmarks_at_landing, R_HEEL, frame_width, frame_height)
    r_toe = _get_landmark_pos(landmarks_at_landing, R_FOOT_INDEX, frame_width, frame_height)
    
    if r_heel is not None and r_toe is not None:
        horizontal_diff = abs(r_toe[0] - r_heel[0])
        vertical_diff = abs(r_toe[1] - r_heel[1])
        front_foot_ratio = horizontal_diff / (vertical_diff + 1e-6)
        signals['front_foot_ratio'] = front_foot_ratio
        
        # Front-on: foot points forward (low ratio), Side-on: foot points sideways (high ratio)
        if front_foot_ratio < 0.5:
            signal_scores['front_foot'] = (0.20, 0.0, 0.0)  # front-on
        else:
            signal_scores['front_foot'] = (0.0, 0.20, 0.0)  # side-on
    
    # Signal 2: Hip alignment (weight 0.25) - KEY for mixed action
    hip_angle = None
    l_hip = _get_landmark_pos(landmarks_at_landing, L_HIP, frame_width, frame_height)
    r_hip = _get_landmark_pos(landmarks_at_landing, R_HIP, frame_width, frame_height)
    
    if l_hip is not None and r_hip is not None:
        hip_angle = _compute_angle_from_points(l_hip, r_hip)
        signals['hip_angle'] = hip_angle
        
        # Hip angle interpretation:
        # 0-20°: front-on (hips level)
        # 20-50°: mixed action (hips slightly rotated)
        # 50-90°: side-on (hips fully rotated to 3 o'clock)
        if hip_angle < 20:
            signal_scores['hip'] = (0.25, 0.0, 0.0)  # front-on
        elif hip_angle < 50:
            signal_scores['hip'] = (0.0, 0.0, 0.25)  # mixed
        else:
            signal_scores['hip'] = (0.0, 0.25, 0.0)  # side-on
    
    # Signal 3: Shoulder alignment (weight 0.25) - KEY for mixed action
    shoulder_angle = None
    l_shoulder = _get_landmark_pos(landmarks_at_landing, L_SHOULDER, frame_width, frame_height)
    r_shoulder = _get_landmark_pos(landmarks_at_landing, R_SHOULDER, frame_width, frame_height)
    
    if l_shoulder is not None and r_shoulder is not None:
        shoulder_angle = _compute_angle_from_points(l_shoulder, r_shoulder)
        signals['shoulder_angle'] = shoulder_angle
        
        # Shoulder angle interpretation:
        # 0-20°: front-on (shoulders level)
        # 20-45°: mixed action (shoulders at 1 o'clock angle)
        # 45-90°: side-on (shoulders fully rotated)
        if shoulder_angle < 20:
            signal_scores['shoulder'] = (0.25, 0.0, 0.0)  # front-on
        elif shoulder_angle < 45:
            signal_scores['shoulder'] = (0.0, 0.0, 0.25)  # mixed
        else:
            signal_scores['shoulder'] = (0.0, 0.25, 0.0)  # side-on
    
    # Signal 4: Back foot angle (weight 0.15)
    back_foot_ratio = None
    l_heel = _get_landmark_pos(landmarks_at_landing, L_HEEL, frame_width, frame_height)
    l_toe = _get_landmark_pos(landmarks_at_landing, L_FOOT_INDEX, frame_width, frame_height)
    
    if l_heel is not None and l_toe is not None:
        horizontal_diff = abs(l_toe[0] - l_heel[0])
        vertical_diff = abs(l_toe[1] - l_heel[1])
        back_foot_ratio = horizontal_diff / (vertical_diff + 1e-6)
        signals['back_foot_ratio'] = back_foot_ratio
        
        if back_foot_ratio < 0.5:
            signal_scores['back_foot'] = (0.15, 0.0, 0.0)  # front-on
        else:
            signal_scores['back_foot'] = (0.0, 0.15, 0.0)  # side-on
    
    # Signal 5: Hip-Shoulder separation (weight 0.15) - CRITICAL for mixed action
    # Mixed action: hips rotated more than shoulders (hips at 3 o'clock, shoulders at 1 o'clock)
    hip_shoulder_separation = None
    if hip_angle is not None and shoulder_angle is not None:
        hip_shoulder_separation = hip_angle - shoulder_angle
        signals['hip_shoulder_separation'] = hip_shoulder_separation
        
        # Positive separation = hips more rotated than shoulders = MIXED action
        # 0-10°: front-on or side-on (aligned)
        # 10-40°: mixed action (hips ahead of shoulders)
        # >40°: extreme mixed (rare)
        if hip_shoulder_separation < 10:
            signal_scores['separation'] = (0.0, 0.0, 0.0)  # neutral
        elif hip_shoulder_separation < 40:
            signal_scores['separation'] = (0.0, 0.0, 0.15)  # mixed
        else:
            signal_scores['separation'] = (0.0, 0.0, 0.075)  # extreme mixed (lower weight)
    
    # Check if we have at least 2 signals
    if len(signal_scores) < 2:
        return "UNKNOWN", {
            'style': 'UNKNOWN',
            'front_score': 0.0,
            'side_score': 0.0,
            'mixed_score': 0.0,
            'signals': signals
        }
    
    # Compute total scores
    total_front_score = sum(score[0] for score in signal_scores.values())
    total_side_score = sum(score[1] for score in signal_scores.values())
    total_mixed_score = sum(score[2] for score in signal_scores.values())
    
    # Normalize scores (they should sum to 1.0)
    total = total_front_score + total_side_score + total_mixed_score
    if total > 0:
        total_front_score /= total
        total_side_score /= total
        total_mixed_score /= total
    
    # Classify with improved thresholds
    # Mixed action gets priority if both hip and shoulder signals agree
    if total_mixed_score >= 0.40:
        style = "MIXED"
    elif total_front_score >= 0.50:
        style = "FRONT_ON"
    elif total_side_score >= 0.50:
        style = "SIDE_ON"
    else:
        style = "MIXED"  # Default to mixed if ambiguous
    
    return style, {
        'style': style,
        'front_score': total_front_score,
        'side_score': total_side_score,
        'mixed_score': total_mixed_score,
        'signals': signals
    }
