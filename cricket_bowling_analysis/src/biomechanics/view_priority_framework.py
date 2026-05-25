"""
src/biomechanics/view_priority_framework.py
---------------------------------------------
View-priority framework for biomechanical feature computation.

Each feature has a primary view (most accurate) and optional secondary views.
This ensures measurements are taken from the most reliable camera angle.

Feature-to-View Mapping:
    SIDE VIEW (Primary for kinematics & joint mechanics):
        - elbow_angle, shoulder_angle, front_knee_angle, back_knee_angle
        - hip_angle, trunk_flexion
    
    FRONT VIEW (Primary for lateral motion & balance):
        - trunk_lean, lateral_flexion
    
    BACK VIEW (Primary for rotation & alignment):
        - hip_shoulder_sep, backfoot_angle, frontfoot_angle
"""

from typing import Optional, Tuple, Dict
from enum import Enum


class ViewType(Enum):
    """Camera view types."""
    SIDE = "side"
    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"


# Feature-to-View Priority Mapping
FEATURE_VIEW_PRIORITY = {
    # SIDE VIEW (Primary for kinematics)
    "elbow_angle": {
        "primary": ViewType.SIDE,
        "secondary": [ViewType.FRONT, ViewType.BACK],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.6,
        "description": "Bowling arm elbow extension - best from side view"
    },
    "shoulder_angle": {
        "primary": ViewType.SIDE,
        "secondary": [ViewType.FRONT, ViewType.BACK],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.65,
        "description": "Shoulder raised relative to body - best from side view"
    },
    "front_knee_angle": {
        "primary": ViewType.SIDE,
        "secondary": [ViewType.FRONT, ViewType.BACK],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.7,
        "description": "Front leg brace at delivery - best from side view"
    },
    "back_knee_angle": {
        "primary": ViewType.SIDE,
        "secondary": [ViewType.FRONT, ViewType.BACK],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.7,
        "description": "Back leg drive - best from side view"
    },
    "hip_angle": {
        "primary": ViewType.SIDE,
        "secondary": [ViewType.FRONT, ViewType.BACK],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.65,
        "description": "Hip position - best from side view"
    },
    "trunk_flexion": {
        "primary": ViewType.SIDE,
        "secondary": [ViewType.FRONT, ViewType.BACK],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.5,
        "description": "Forward/backward lean in sagittal plane - best from side view"
    },
    
    # FRONT VIEW (Primary for lateral motion)
    "trunk_lean": {
        "primary": ViewType.FRONT,
        "secondary": [ViewType.SIDE, ViewType.BACK],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.5,
        "description": "Lateral tilt of torso - best from front view"
    },
    "lateral_flexion": {
        "primary": ViewType.FRONT,
        "secondary": [ViewType.SIDE, ViewType.BACK],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.55,
        "description": "Lateral bend of torso at release - best from front view"
    },
    
    # BACK VIEW (Primary for rotation & alignment)
    "hip_shoulder_sep": {
        "primary": ViewType.BACK,
        "secondary": [ViewType.SIDE],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.4,
        "description": "Hip-shoulder separation (rotation) - CRITICAL: back view only",
        "strict_primary": True
    },
    "backfoot_angle": {
        "primary": ViewType.BACK,
        "secondary": [ViewType.SIDE],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.45,
        "description": "Back foot angle - CRITICAL: back view only",
        "strict_primary": True
    },
    "frontfoot_angle": {
        "primary": ViewType.BACK,
        "secondary": [ViewType.SIDE],
        "confidence_primary": 1.0,
        "confidence_secondary": 0.45,
        "description": "Front foot angle - CRITICAL: back view only",
        "strict_primary": True
    },
}


def get_feature_priority(feature_name: str) -> Optional[Dict]:
    """
    Get the view priority mapping for a feature.
    
    Parameters
    ----------
    feature_name : str
        Name of the biomechanical feature
    
    Returns
    -------
    dict or None
        Priority mapping with primary view, secondary views, and confidence scores
    """
    return FEATURE_VIEW_PRIORITY.get(feature_name)


def get_confidence_score(feature_name: str, view_used: ViewType) -> float:
    """
    Get the confidence score for a feature computed from a specific view.
    
    Parameters
    ----------
    feature_name : str
        Name of the biomechanical feature
    view_used : ViewType
        The view from which the feature was computed
    
    Returns
    -------
    float
        Confidence score (0.0-1.0). Returns 0.0 if view is not valid for feature.
    """
    priority = get_feature_priority(feature_name)
    if priority is None:
        return 0.0
    
    if view_used == priority["primary"]:
        return priority["confidence_primary"]
    
    if view_used in priority["secondary"]:
        return priority["confidence_secondary"]
    
    return 0.0


def is_strict_primary_required(feature_name: str) -> bool:
    """
    Check if a feature MUST use its primary view (no fallback allowed).
    
    Parameters
    ----------
    feature_name : str
        Name of the biomechanical feature
    
    Returns
    -------
    bool
        True if feature requires strict primary view usage
    """
    priority = get_feature_priority(feature_name)
    if priority is None:
        return False
    return priority.get("strict_primary", False)


def get_primary_view(feature_name: str) -> Optional[ViewType]:
    """
    Get the primary (most accurate) view for a feature.
    
    Parameters
    ----------
    feature_name : str
        Name of the biomechanical feature
    
    Returns
    -------
    ViewType or None
        Primary view for the feature
    """
    priority = get_feature_priority(feature_name)
    if priority is None:
        return None
    return priority["primary"]


def get_secondary_views(feature_name: str) -> list:
    """
    Get the secondary (fallback) views for a feature.
    
    Parameters
    ----------
    feature_name : str
        Name of the biomechanical feature
    
    Returns
    -------
    list of ViewType
        Secondary views in priority order
    """
    priority = get_feature_priority(feature_name)
    if priority is None:
        return []
    return priority.get("secondary", [])


def select_best_view(feature_name: str, available_views: list) -> Tuple[Optional[ViewType], float]:
    """
    Select the best available view for computing a feature.
    
    Parameters
    ----------
    feature_name : str
        Name of the biomechanical feature
    available_views : list of ViewType
        List of available camera views
    
    Returns
    -------
    tuple (ViewType or None, float)
        (best_view, confidence_score)
        Returns (None, 0.0) if no valid view is available
    """
    priority = get_feature_priority(feature_name)
    if priority is None:
        return None, 0.0
    
    # Check if primary view is available
    if priority["primary"] in available_views:
        return priority["primary"], priority["confidence_primary"]
    
    # If strict primary is required, return None
    if priority.get("strict_primary", False):
        return None, 0.0
    
    # Check secondary views in order
    for secondary_view in priority.get("secondary", []):
        if secondary_view in available_views:
            return secondary_view, priority["confidence_secondary"]
    
    return None, 0.0


def create_feature_metadata(feature_name: str, view_used: ViewType, 
                           value: float) -> Dict:
    """
    Create metadata for a computed feature including confidence and view info.
    
    Parameters
    ----------
    feature_name : str
        Name of the biomechanical feature
    view_used : ViewType
        The view from which the feature was computed
    value : float
        The computed feature value
    
    Returns
    -------
    dict
        Metadata dict with value, view, confidence, and description
    """
    priority = get_feature_priority(feature_name)
    confidence = get_confidence_score(feature_name, view_used)
    
    return {
        "value": value,
        "view": view_used.value,
        "confidence": confidence,
        "is_primary_view": view_used == priority["primary"] if priority else False,
        "description": priority["description"] if priority else "Unknown feature",
        "strict_primary_required": is_strict_primary_required(feature_name)
    }


def validate_feature_for_view(feature_name: str, view: ViewType) -> Tuple[bool, str]:
    """
    Validate if a feature can be computed from a specific view.
    
    Parameters
    ----------
    feature_name : str
        Name of the biomechanical feature
    view : ViewType
        The camera view to validate
    
    Returns
    -------
    tuple (bool, str)
        (is_valid, message)
    """
    priority = get_feature_priority(feature_name)
    
    if priority is None:
        return False, f"Unknown feature: {feature_name}"
    
    if view == priority["primary"]:
        return True, f"Primary view for {feature_name}"
    
    if view in priority["secondary"]:
        confidence = priority["confidence_secondary"]
        return True, f"Secondary view for {feature_name} (confidence: {confidence})"
    
    if priority.get("strict_primary", False):
        return False, f"{feature_name} requires primary view ({priority['primary'].value})"
    
    return False, f"View {view.value} not suitable for {feature_name}"


# View mapping for multi-camera sessions
VIEW_MAPPING = {
    "front": ViewType.FRONT,
    "back": ViewType.BACK,
    "left": ViewType.SIDE,
    "right": ViewType.SIDE,
}


def get_view_type_from_camera_name(camera_name: str) -> Optional[ViewType]:
    """
    Convert camera name to ViewType.
    
    Parameters
    ----------
    camera_name : str
        Camera name (e.g., 'front', 'back', 'left', 'right')
    
    Returns
    -------
    ViewType or None
        Corresponding ViewType
    """
    return VIEW_MAPPING.get(camera_name.lower())
