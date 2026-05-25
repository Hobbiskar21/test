"""
src/biomechanics/angle_calculator.py
--------------------------------------
Computes joint angles from COCO keypoints (17 points) per frame.
Uses view-priority framework to ensure measurements are taken from optimal camera angles.

COCO Keypoint Indices:
0=nose, 5=lsho, 6=rsho, 7=lelb, 8=relb, 9=lwri, 10=rwri,
11=lhip, 12=rhip, 13=lknee, 14=rknee, 15=lank, 16=rank

Angles computed (all in degrees, 0-180):
    elbow_angle        : bowling arm elbow extension
    shoulder_angle     : bowling arm raised relative to body
    front_knee_angle   : front leg brace at delivery
    back_knee_angle    : back leg drive
    hip_angle          : hip position
    trunk_lean         : lateral tilt of torso
    trunk_flexion      : forward/backward lean in sagittal plane (0d=upright, 90d=fully bent)
    hip_shoulder_sep   : angle between shoulder line and hip line
    lateral_flexion    : lateral bend of torso at release (nose-hip-shoulder)
    backfoot_angle     : angle of back foot (left foot for right-arm bowler)
    frontfoot_angle    : angle of front foot (right foot for right-arm bowler)

To add a new angle:
    1. Extract the 3 joint landmarks using COCO indices
    2. Call calculate_angle(a, b, c)
    3. Add result to angles dict
    4. Add view priority mapping in view_priority_framework.py
"""

import numpy as np
from typing import Optional
from src.biomechanics.view_priority_framework import (
    get_feature_priority, get_confidence_score, ViewType
)


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    """
    Angle in degrees at joint B formed by A → B → C.
    
    Returns float between 0-180, or None if calculation fails.
    Ensures result is always a proper float or None (never NaN, Inf, or string).
    """
    try:
        # Validate inputs are numpy arrays
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray) or not isinstance(c, np.ndarray):
            return None
        
        ba     = a - b
        bc     = c - b
        
        # Calculate norms
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        # Check for zero-length vectors (invalid)
        if norm_ba < 1e-8 or norm_bc < 1e-8:
            return None
        
        # Check for NaN or Inf in norms
        if np.isnan(norm_ba) or np.isinf(norm_ba) or np.isnan(norm_bc) or np.isinf(norm_bc):
            return None
        
        # Calculate cosine with clipping to [-1, 1]
        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        
        # Check for NaN or Inf in cosine
        if np.isnan(cosine) or np.isinf(cosine):
            return None
        
        cosine = np.clip(cosine, -1.0, 1.0)
        
        # Calculate angle
        angle_rad = np.arccos(cosine)
        angle_deg = float(np.degrees(angle_rad))
        
        # Final validation: ensure result is a valid float
        if not isinstance(angle_deg, float):
            return None
        if np.isnan(angle_deg) or np.isinf(angle_deg):
            return None
        if angle_deg < 0 or angle_deg > 180:
            return None
        
        return angle_deg
    except (ValueError, TypeError, RuntimeError, AttributeError):
        return None


def get_landmark(landmarks: list, index: int, width: int, height: int,
                 min_confidence: float = 0.5) -> Optional[np.ndarray]:
    """Extract one landmark as pixel-space numpy array. Returns None if low confidence."""
    lm = landmarks[index]
    if lm.visibility < min_confidence:
        return None
    return np.array([lm.x * width, lm.y * height])


def compute_all_angles(landmarks: list, width: int, height: int,
                       min_confidence: float = 0.5) -> dict:
    """
    Compute all biomechanical angles for a single frame.
    Works with COCO keypoints (17 points).
    
    COCO indices:
    0=nose, 5=lsho, 6=rsho, 7=lelb, 8=relb, 9=lwri, 10=rwri,
    11=lhip, 12=rhip, 13=lknee, 14=rknee, 15=lank, 16=rank

    Returns:
        dict of angle_name -> float (degrees) or None if landmarks missing.
    """
    def get(idx):
        return get_landmark(landmarks, idx, width, height, min_confidence)

    # Check overall pose quality (detect person switches)
    visible_landmarks = sum(1 for lm in landmarks if lm.visibility > min_confidence)
    if visible_landmarks < 10:
        # Less than half the landmarks visible - might be wrong person or bad frame
        pass  # Still compute angles, but they may be unreliable
    
    # COCO keypoint indices
    nose       = get(0)
    l_shoulder = get(5)
    r_shoulder = get(6)
    l_elbow    = get(7)
    r_elbow    = get(8)
    l_wrist    = get(9)
    r_wrist    = get(10)
    l_hip      = get(11)
    r_hip      = get(12)
    l_knee     = get(13)
    r_knee     = get(14)
    l_ankle    = get(15)
    r_ankle    = get(16)
    # COCO doesn't have heel/foot_idx, use ankle as reference
    l_heel     = get(15)  # left ankle
    r_heel     = get(16)  # right ankle
    l_foot_idx = get(15)  # left ankle
    r_foot_idx = get(16)  # right ankle

    mid_hip = None
    if l_hip is not None and r_hip is not None:
        mid_hip = (l_hip + r_hip) / 2.0

    angles = {}

    # Bowling arm elbow (right arm)
    if r_shoulder is not None and r_elbow is not None and r_wrist is not None:
        angles["elbow_angle"] = calculate_angle(r_shoulder, r_elbow, r_wrist)
    else:
        angles["elbow_angle"] = None

    # Shoulder angle
    if nose is not None and r_shoulder is not None and r_elbow is not None:
        angles["shoulder_angle"] = calculate_angle(nose, r_shoulder, r_elbow)
    else:
        angles["shoulder_angle"] = None

    # Front knee (left knee for right-arm bowler)
    if l_hip is not None and l_knee is not None and l_ankle is not None:
        angles["front_knee_angle"] = calculate_angle(l_hip, l_knee, l_ankle)
    else:
        angles["front_knee_angle"] = None

    # Back knee (right knee for right-arm bowler)
    if r_hip is not None and r_knee is not None and r_ankle is not None:
        angles["back_knee_angle"] = calculate_angle(r_hip, r_knee, r_ankle)
    else:
        angles["back_knee_angle"] = None

    # Hip angle
    if nose is not None and r_hip is not None and r_knee is not None:
        angles["hip_angle"] = calculate_angle(nose, r_hip, r_knee)
    else:
        angles["hip_angle"] = None

    # Trunk lean
    if nose is not None and mid_hip is not None and r_hip is not None:
        angles["trunk_lean"] = calculate_angle(nose, mid_hip, r_hip)
    else:
        angles["trunk_lean"] = None

    # Hip-shoulder separation
    if (l_shoulder is not None and r_shoulder is not None and
            l_hip is not None and r_hip is not None):
        try:
            shoulder_line = r_shoulder - l_shoulder
            hip_line      = r_hip - l_hip
            
            # Check for NaN or Inf in vectors
            if np.any(np.isnan(shoulder_line)) or np.any(np.isinf(shoulder_line)):
                angles["hip_shoulder_sep"] = None
            elif np.any(np.isnan(hip_line)) or np.any(np.isinf(hip_line)):
                angles["hip_shoulder_sep"] = None
            else:
                norm_shoulder = np.linalg.norm(shoulder_line)
                norm_hip = np.linalg.norm(hip_line)
                
                # Check for NaN or Inf in norms
                if np.isnan(norm_shoulder) or np.isinf(norm_shoulder) or np.isnan(norm_hip) or np.isinf(norm_hip):
                    angles["hip_shoulder_sep"] = None
                # Check for zero-length vectors
                elif norm_shoulder < 1e-8 or norm_hip < 1e-8:
                    angles["hip_shoulder_sep"] = None
                else:
                    cos_sep = np.dot(shoulder_line, hip_line) / (norm_shoulder * norm_hip)
                    
                    # Check for NaN or Inf in cosine
                    if np.isnan(cos_sep) or np.isinf(cos_sep):
                        angles["hip_shoulder_sep"] = None
                    else:
                        cos_sep = np.clip(cos_sep, -1.0, 1.0)
                        angle_deg = float(np.degrees(np.arccos(cos_sep)))
                        
                        # Validate result
                        if not isinstance(angle_deg, float):
                            angles["hip_shoulder_sep"] = None
                        elif np.isnan(angle_deg) or np.isinf(angle_deg) or angle_deg < 0 or angle_deg > 180:
                            angles["hip_shoulder_sep"] = None
                        else:
                            angles["hip_shoulder_sep"] = angle_deg
        except (ValueError, TypeError, RuntimeError, AttributeError):
            angles["hip_shoulder_sep"] = None
    else:
        angles["hip_shoulder_sep"] = None

    # Lateral flexion angle (lateral bend at release)
    if nose is not None and mid_hip is not None and r_shoulder is not None:
        angles["lateral_flexion"] = calculate_angle(nose, mid_hip, r_shoulder)
    else:
        angles["lateral_flexion"] = None

    # Trunk flexion (forward/backward lean in sagittal plane)
    # Measured as angle between vertical line and nose-hip line
    # 0d = upright, 90d = fully bent forward
    if nose is not None and mid_hip is not None:
        try:
            # Vector from hip to nose (upward direction)
            trunk_vector = nose - mid_hip
            
            # Check for NaN or Inf in vector
            if np.any(np.isnan(trunk_vector)) or np.any(np.isinf(trunk_vector)):
                angles["trunk_flexion"] = None
            else:
                # Vertical reference vector (pointing up in image space)
                vertical = np.array([0, -1])
                
                norm_trunk = np.linalg.norm(trunk_vector)
                norm_vertical = np.linalg.norm(vertical)
                
                # Check for zero-length vectors
                if norm_trunk < 1e-8 or norm_vertical < 1e-8:
                    angles["trunk_flexion"] = None
                else:
                    cos_flex = np.dot(trunk_vector, vertical) / (norm_trunk * norm_vertical)
                    
                    # Check for NaN or Inf in cosine
                    if np.isnan(cos_flex) or np.isinf(cos_flex):
                        angles["trunk_flexion"] = None
                    else:
                        cos_flex = np.clip(cos_flex, -1.0, 1.0)
                        angle_deg = float(np.degrees(np.arccos(cos_flex)))
                        
                        # Validate result
                        if not isinstance(angle_deg, float):
                            angles["trunk_flexion"] = None
                        elif np.isnan(angle_deg) or np.isinf(angle_deg) or angle_deg < 0 or angle_deg > 180:
                            angles["trunk_flexion"] = None
                        else:
                            angles["trunk_flexion"] = angle_deg
        except (ValueError, TypeError, RuntimeError, AttributeError):
            angles["trunk_flexion"] = None
    else:
        angles["trunk_flexion"] = None

    # Backfoot angle (left foot for right-arm bowler)
    # Angle between heel and toe of back foot
    if l_heel is not None and l_foot_idx is not None:
        try:
            # Vector from heel to foot index (toe direction)
            foot_vector = l_foot_idx - l_heel
            
            # Check for NaN or Inf
            if np.any(np.isnan(foot_vector)) or np.any(np.isinf(foot_vector)):
                angles["backfoot_angle"] = None
            else:
                # Horizontal reference (pointing right)
                horizontal = np.array([1, 0])
                
                norm_foot = np.linalg.norm(foot_vector)
                norm_horizontal = np.linalg.norm(horizontal)
                
                if norm_foot < 1e-8 or norm_horizontal < 1e-8:
                    angles["backfoot_angle"] = None
                else:
                    cos_angle = np.dot(foot_vector, horizontal) / (norm_foot * norm_horizontal)
                    
                    if np.isnan(cos_angle) or np.isinf(cos_angle):
                        angles["backfoot_angle"] = None
                    else:
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_deg = float(np.degrees(np.arccos(cos_angle)))
                        
                        if not isinstance(angle_deg, float):
                            angles["backfoot_angle"] = None
                        elif np.isnan(angle_deg) or np.isinf(angle_deg) or angle_deg < 0 or angle_deg > 180:
                            angles["backfoot_angle"] = None
                        else:
                            angles["backfoot_angle"] = angle_deg
        except (ValueError, TypeError, RuntimeError, AttributeError):
            angles["backfoot_angle"] = None
    else:
        angles["backfoot_angle"] = None

    # Frontfoot angle (right foot for right-arm bowler)
    # Angle between heel and toe of front foot
    if r_heel is not None and r_foot_idx is not None:
        try:
            # Vector from heel to foot index (toe direction)
            foot_vector = r_foot_idx - r_heel
            
            # Check for NaN or Inf
            if np.any(np.isnan(foot_vector)) or np.any(np.isinf(foot_vector)):
                angles["frontfoot_angle"] = None
            else:
                # Horizontal reference (pointing right)
                horizontal = np.array([1, 0])
                
                norm_foot = np.linalg.norm(foot_vector)
                norm_horizontal = np.linalg.norm(horizontal)
                
                if norm_foot < 1e-8 or norm_horizontal < 1e-8:
                    angles["frontfoot_angle"] = None
                else:
                    cos_angle = np.dot(foot_vector, horizontal) / (norm_foot * norm_horizontal)
                    
                    if np.isnan(cos_angle) or np.isinf(cos_angle):
                        angles["frontfoot_angle"] = None
                    else:
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_deg = float(np.degrees(np.arccos(cos_angle)))
                        
                        if not isinstance(angle_deg, float):
                            angles["frontfoot_angle"] = None
                        elif np.isnan(angle_deg) or np.isinf(angle_deg) or angle_deg < 0 or angle_deg > 180:
                            angles["frontfoot_angle"] = None
                        else:
                            angles["frontfoot_angle"] = angle_deg
        except (ValueError, TypeError, RuntimeError, AttributeError):
            angles["frontfoot_angle"] = None
    else:
        angles["frontfoot_angle"] = None

    return angles


def summarize_angle_sequence(angle_sequence: list) -> dict:
    """
    Compute min, max, mean for each angle across all frames.

    Returns:
        {angle_name: {"min": float, "max": float, "mean": float}}
    """
    from collections import defaultdict
    buckets = defaultdict(list)

    for frame_angles in angle_sequence:
        for name, val in frame_angles.items():
            if val is not None:
                buckets[name].append(val)

    summary = {}
    for name, values in buckets.items():
        arr = np.array(values)
        summary[name] = {
            "min":  round(float(arr.min()), 2),
            "max":  round(float(arr.max()), 2),
            "mean": round(float(arr.mean()), 2),
        }
    return summary


def compute_angles_with_view_priority(landmarks: list, width: int, height: int,
                                      camera_view: str = "side",
                                      min_confidence: float = 0.5) -> tuple[dict, dict]:
    """
    Compute all angles with view-priority framework tracking.
    
    Parameters
    ----------
    landmarks : list
        COCO landmarks (17 keypoints) for a single frame
    width : int
        Frame width in pixels
    height : int
        Frame height in pixels
    camera_view : str
        Camera view name ('front', 'back', 'left', 'right', 'side')
    min_confidence : float
        Minimum landmark visibility threshold
    
    Returns
    -------
    tuple (angles_dict, metadata_dict)
        angles_dict: {angle_name: float or None}
        metadata_dict: {angle_name: {value, view, confidence, is_primary_view, ...}}
    """
    from src.biomechanics.view_priority_framework import get_view_type_from_camera_name
    
    # Get view type from camera name
    view_type = get_view_type_from_camera_name(camera_view)
    if view_type is None:
        view_type = ViewType.SIDE  # Default to side view
    
    # Compute all angles
    angles = compute_all_angles(landmarks, width, height, min_confidence)
    
    # Create metadata for each angle
    metadata = {}
    for angle_name, angle_value in angles.items():
        if angle_value is not None:
            priority = get_feature_priority(angle_name)
            if priority:
                confidence = get_confidence_score(angle_name, view_type)
                is_primary = view_type == priority["primary"]
                
                metadata[angle_name] = {
                    "value": angle_value,
                    "view": view_type.value,
                    "confidence": confidence,
                    "is_primary_view": is_primary,
                    "description": priority["description"],
                    "strict_primary_required": priority.get("strict_primary", False)
                }
            else:
                metadata[angle_name] = {
                    "value": angle_value,
                    "view": view_type.value,
                    "confidence": 0.5,
                    "is_primary_view": False,
                    "description": "Unknown feature",
                    "strict_primary_required": False
                }
        else:
            metadata[angle_name] = {
                "value": None,
                "view": view_type.value,
                "confidence": 0.0,
                "is_primary_view": False,
                "description": "Angle computation failed",
                "strict_primary_required": False
            }
    
    return angles, metadata
