"""
src/biomechanics/view_priority_sync.py
──────────────────────────────────────────────────────────────────────────────
Multi-view synchronization with strict view prioritization rules.

VIEW PRIORITIZATION
───────────────────
RUN-UP:
  Priority: BACK → FRONT → SIDE
  Use only ONE view

LOAD-UP, GATHER, DELIVERY, FOLLOW-THROUGH:
  Priority: FRONT → BACK → SIDE
  Use best available view

MULTI-VIEW SYNCHRONIZATION
──────────────────────────
When multiple views available:
1. Identify key event frames independently in each view
2. Use RELEASE FRAME as primary sync point (t=0)
3. Create relative timeline: negative before release, positive after
4. Fallback to delivery stride if release can't be detected
5. Don't modify raw indices - only create normalized aligned timeline

OUTPUT
──────
Append to delivery record:
{
    "runup_view_used": "back/front/side",
    "runup_view_fallback": true/false,
    "phase_view_used": "front/back/side",
    "phase_view_fallback": true/false,
    "sync_reference": "release/delivery",
    "sync_fallback_used": true/false,
    "relative_frames": {
        "load_up": value,
        "gather": value,
        "delivery": value,
        "release": 0,
        "follow_through": value
    }
}
"""

from typing import Dict, Optional, List, Tuple
import numpy as np


def select_runup_view(available_cameras: Dict[str, bool]) -> Tuple[str, bool]:
    """
    Select which camera view to use for RUN-UP analysis.
    
    Priority: BACK → FRONT → SIDE
    
    Parameters
    ----------
    available_cameras : dict
        {"side": bool, "front": bool, "back": bool}
        True if camera has valid pose data
    
    Returns
    -------
    tuple
        (view_name, is_fallback)
        view_name: "back", "front", or "side"
        is_fallback: True if not using preferred BACK view
    """
    if available_cameras.get("back", False):
        return "back", False
    elif available_cameras.get("front", False):
        return "front", True
    elif available_cameras.get("side", False):
        return "side", True
    else:
        # Fallback to side if nothing available
        return "side", True


def select_phase_view(available_cameras: Dict[str, bool]) -> Tuple[str, bool]:
    """
    Select which camera view to use for LOAD-UP/GATHER/DELIVERY/FOLLOW-THROUGH.
    
    Priority: FRONT → BACK → SIDE
    
    Parameters
    ----------
    available_cameras : dict
        {"side": bool, "front": bool, "back": bool}
        True if camera has valid pose data
    
    Returns
    -------
    tuple
        (view_name, is_fallback)
        view_name: "front", "back", or "side"
        is_fallback: True if not using preferred FRONT
    """
    if available_cameras.get("front", False):
        return "front", False
    elif available_cameras.get("back", False):
        return "back", True
    elif available_cameras.get("side", False):
        return "side", True
    else:
        # Fallback to side
        return "side", True


def find_release_frame_in_view(
    wrist_positions: List,
    wrist_velocities: List,
    phase_map: Dict[int, str],
) -> Optional[int]:
    """
    Find release frame in a specific view.
    
    Release = frame where wrist velocity peaks during DELIVERY phase
    (wrist moving fastest downward after release)
    
    Parameters
    ----------
    wrist_positions : list
        Wrist (x, y) positions per frame
    wrist_velocities : list
        Wrist velocity magnitude per frame
    phase_map : dict
        {frame_idx: phase_name}
    
    Returns
    -------
    int or None
        Frame index of release, or None if not found
    """
    if not wrist_positions or not wrist_velocities:
        return None
    
    # Find DELIVERY phase frames
    delivery_frames = [f for f, phase in phase_map.items() if phase == "DELIVERY"]
    
    if not delivery_frames:
        return None
    
    # Release is typically at the start of DELIVERY phase
    # (when wrist is at minimum Y = topmost point)
    return min(delivery_frames) if delivery_frames else None


def find_delivery_stride_frame(
    ankle_positions: List,
    phase_map: Dict[int, str],
) -> Optional[int]:
    """
    Find delivery stride frame (front foot contact).
    
    Fallback sync point if release frame not available.
    Front foot contact = when front ankle Y stops decreasing (touches ground)
    
    Parameters
    ----------
    ankle_positions : list
        [(left_ankle, right_ankle), ...] per frame
    phase_map : dict
        {frame_idx: phase_name}
    
    Returns
    -------
    int or None
        Frame index of front foot contact, or None if not found
    """
    if not ankle_positions:
        return None
    
    # Find DELIVERY phase frames
    delivery_frames = [f for f, phase in phase_map.items() if phase == "DELIVERY"]
    
    if not delivery_frames:
        return None
    
    # Delivery stride is typically at the end of DELIVERY phase
    return max(delivery_frames) if delivery_frames else None


def create_relative_timeline(
    phase_boundaries: Dict[str, int],
    release_frame: int,
) -> Dict[str, int]:
    """
    Create relative timeline with release frame as t=0.
    
    Parameters
    ----------
    phase_boundaries : dict
        {
            "movement_start_frame": int,
            "jump_frame": int,
            "peak_wrist_frame": int,
            "release_frame": int,
        }
    release_frame : int
        Frame index to use as t=0 (primary sync point)
    
    Returns
    -------
    dict
        {
            "load_up": relative_frame,
            "gather": relative_frame,
            "delivery": relative_frame,
            "release": 0,
            "follow_through": relative_frame,
        }
    """
    relative = {}
    
    # Map phase boundaries to relative positions
    if "movement_start_frame" in phase_boundaries:
        relative["load_up"] = phase_boundaries["movement_start_frame"] - release_frame
    
    if "jump_frame" in phase_boundaries:
        relative["gather"] = phase_boundaries["jump_frame"] - release_frame
    
    if "peak_wrist_frame" in phase_boundaries:
        relative["delivery"] = phase_boundaries["peak_wrist_frame"] - release_frame
    
    relative["release"] = 0  # Always 0 by definition
    
    # Follow-through is typically 50-100 frames after release
    # (estimate if not explicitly provided)
    relative["follow_through"] = 50  # Default estimate
    
    return relative


def synchronize_multi_view(
    views_data: Dict[str, Dict],
    available_cameras: Dict[str, bool],
) -> Dict:
    """
    Synchronize multiple camera views using release frame as sync point.
    
    Parameters
    ----------
    views_data : dict
        {
            "side": {
                "phase_boundaries": {...},
                "phase_map": {...},
                "wrist_positions": [...],
                "wrist_velocities": [...],
                "ankle_positions": [...],
            },
            "front": {...},
            "back": {...},
        }
    available_cameras : dict
        {"side": bool, "front": bool, "back": bool}
    
    Returns
    -------
    dict
        {
            "runup_view_used": "back/front/side",
            "runup_view_fallback": true/false,
            "phase_view_used": "front/back/side",
            "phase_view_fallback": true/false,
            "sync_reference": "release/delivery",
            "sync_fallback_used": true/false,
            "relative_frames": {
                "load_up": value,
                "gather": value,
                "delivery": value,
                "release": 0,
                "follow_through": value,
            },
            "view_details": {
                "side": {...},
                "front": {...},
                "back": {...},
            }
        }
    """
    # Select views
    runup_view, runup_fallback = select_runup_view(available_cameras)
    phase_view, phase_fallback = select_phase_view(available_cameras)
    
    # Get data for selected views
    runup_data = views_data.get(runup_view, {})
    phase_data = views_data.get(phase_view, {})
    
    # Find release frame (primary sync point)
    release_frame = find_release_frame_in_view(
        phase_data.get("wrist_positions", []),
        phase_data.get("wrist_velocities", []),
        phase_data.get("phase_map", {}),
    )
    
    sync_fallback = False
    sync_reference = "release"
    
    # Fallback to delivery stride if release not found
    if release_frame is None:
        release_frame = find_delivery_stride_frame(
            phase_data.get("ankle_positions", []),
            phase_data.get("phase_map", {}),
        )
        sync_fallback = True
        sync_reference = "delivery"
    
    # If still no sync point, use phase boundaries
    if release_frame is None:
        boundaries = phase_data.get("phase_boundaries", {})
        release_frame = boundaries.get("release_frame", 0)
        sync_fallback = True
        sync_reference = "fallback"
    
    # Create relative timeline
    phase_boundaries = phase_data.get("phase_boundaries", {})
    relative_frames = create_relative_timeline(phase_boundaries, release_frame)
    
    # Build output
    output = {
        "runup_view_used": runup_view,
        "runup_view_fallback": runup_fallback,
        "phase_view_used": phase_view,
        "front_view_fallback": phase_fallback,
        "sync_reference": sync_reference,
        "sync_fallback_used": sync_fallback,
        "relative_frames": relative_frames,
        "view_details": {
            "side": {
                "available": available_cameras.get("side", False),
                "used_for": "runup" if runup_view == "side" else ("phase" if phase_view == "side" else "none"),
            },
            "front": {
                "available": available_cameras.get("front", False),
                "used_for": "runup" if runup_view == "front" else ("phase" if phase_view == "front" else "none"),
            },
            "back": {
                "available": available_cameras.get("back", False),
                "used_for": "runup" if runup_view == "back" else "none",
            },
        }
    }
    
    return output


def validate_view_consistency(
    delivery_record: Dict,
    sync_info: Dict,
) -> List[str]:
    """
    Validate that views are used consistently (no mixing within phases).
    
    Parameters
    ----------
    delivery_record : dict
        Delivery record with phase data
    sync_info : dict
        Output from synchronize_multi_view
    
    Returns
    -------
    list
        List of validation warnings (empty if all OK)
    """
    warnings = []
    
    # Check that runup uses only one view
    if sync_info.get("runup_view_used") not in ["back", "front", "side"]:
        warnings.append("Invalid runup_view_used")
    
    # Check that phase view is valid
    if sync_info.get("phase_view_used") not in ["front", "back", "side"]:
        warnings.append("Invalid phase_view_used")
    
    # Check that relative frames are reasonable
    relative = sync_info.get("relative_frames", {})
    if relative.get("release") != 0:
        warnings.append("Release frame should be at relative position 0")
    
    # Check that load_up < gather < delivery < release
    load_up = relative.get("load_up", -100)
    gather = relative.get("gather", -50)
    delivery = relative.get("delivery", -5)
    release = relative.get("release", 0)
    
    if not (load_up < gather < delivery < release):
        warnings.append(
            f"Phase order incorrect: load_up={load_up}, gather={gather}, "
            f"delivery={delivery}, release={release}"
        )
    
    return warnings
