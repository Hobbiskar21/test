"""
src/biomechanics/feature_aggregator.py
----------------------------------------
Combines all biomechanics outputs into a single flat DeliveryRecord dict.
This is the one object passed to storage (CSV, Parquet, or PostgreSQL).

To add a new feature:
    1. Compute it anywhere in the pipeline
    2. Pass it into build_delivery_record()
    3. Add the key to the returned dict
"""

from typing import Optional
from src.biomechanics.view_priority_framework import (
    get_primary_view, is_strict_primary_required, ViewType, get_view_type_from_camera_name
)


def build_delivery_record(session_id: str,
                           delivery_number: int,
                           bowler_name: str,
                           fps: float,
                           phase_map: dict,
                           phase_ranges: dict,
                           release_frame: Optional[int],
                           angles_at_release: dict,
                           angle_summary: dict,
                           velocities: dict,
                           bowling_style: Optional[str] = None,
                           bowling_style_breakdown: Optional[dict] = None,
                           runup_metrics: Optional[dict] = None,
                           angle_metadata: Optional[dict] = None,
                           camera_view: Optional[str] = None,
                           view_sync_info: Optional[dict] = None) -> dict:
    """
    Build the complete DeliveryRecord for one delivery.
    
    Only stores angle data when computed from the correct primary view.
    If camera_view doesn't match the feature's primary view, angle is set to None.
    
    Parameters
    ----------
    angle_metadata : dict, optional
        Metadata for angles including view and confidence scores
    camera_view : str, optional
        Camera view used for analysis (e.g., 'front', 'back', 'side')
    view_sync_info : dict, optional
        Multi-view synchronization info from view_priority_sync module:
        {
            "runup_view_used": str,
            "runup_view_fallback": bool,
            "phase_view_used": str,
            "front_view_fallback": bool,
            "sync_reference": str,
            "sync_fallback_used": bool,
            "relative_frames": dict,
        }

    Returns:
        Flat dict ready for CSV, Parquet, or PostgreSQL.
    """
    if bowling_style_breakdown is None:
        bowling_style_breakdown = {}
    if runup_metrics is None:
        runup_metrics = {}
    if angle_metadata is None:
        angle_metadata = {}
    if view_sync_info is None:
        view_sync_info = {}
    
    # Get the view type from camera name
    current_view = get_view_type_from_camera_name(camera_view) if camera_view else ViewType.SIDE
    
    # Helper function to get angle value only if from correct primary view
    def get_angle_if_primary(angle_name: str, angle_value: Optional[float]) -> Optional[float]:
        """Return angle value only if computed from its primary view."""
        if angle_value is None:
            return None
        
        primary_view = get_primary_view(angle_name)
        if primary_view is None:
            return angle_value  # Unknown feature, store anyway
        
        # Only store if current view matches primary view
        if current_view == primary_view:
            return angle_value
        
        # For strict primary features, never store from secondary view
        if is_strict_primary_required(angle_name):
            return None
        
        # For flexible features, allow secondary views with reduced confidence
        # But still mark as None to indicate it's not from primary view
        return None
    
    def get_confidence_if_primary(angle_name: str) -> Optional[float]:
        """Return confidence only if from primary view."""
        primary_view = get_primary_view(angle_name)
        if primary_view is None:
            return None
        
        if current_view == primary_view:
            return angle_metadata.get(angle_name, {}).get("confidence")
        
        return None
    
    return {
        # Identity
        "session_id":       session_id,
        "delivery_number":  delivery_number,
        "bowler_name":      bowler_name,
        "camera_view":      camera_view,

        # Release
        "release_frame":        release_frame,

        # Angles at release frame - ONLY stored if from primary view
        "elbow_angle_at_release":       get_angle_if_primary("elbow_angle", angles_at_release.get("elbow_angle")),
        "shoulder_angle_at_release":    get_angle_if_primary("shoulder_angle", angles_at_release.get("shoulder_angle")),
        "front_knee_angle_at_release":  get_angle_if_primary("front_knee_angle", angles_at_release.get("front_knee_angle")),
        "back_knee_angle_at_release":   get_angle_if_primary("back_knee_angle", angles_at_release.get("back_knee_angle")),
        "hip_angle_at_release":         get_angle_if_primary("hip_angle", angles_at_release.get("hip_angle")),
        "trunk_lean_at_release":        get_angle_if_primary("trunk_lean", angles_at_release.get("trunk_lean")),
        "trunk_flexion_at_release":     get_angle_if_primary("trunk_flexion", angles_at_release.get("trunk_flexion")),
        "hip_shoulder_sep_at_release":  get_angle_if_primary("hip_shoulder_sep", angles_at_release.get("hip_shoulder_sep")),
        "lateral_flexion_at_release":   get_angle_if_primary("lateral_flexion", angles_at_release.get("lateral_flexion")),
        "backfoot_angle_at_release":    get_angle_if_primary("backfoot_angle", angles_at_release.get("backfoot_angle")),
        "frontfoot_angle_at_release":   get_angle_if_primary("frontfoot_angle", angles_at_release.get("frontfoot_angle")),

        # Peak angles across full delivery - ONLY stored if from primary view
        "elbow_angle_max":       get_angle_if_primary("elbow_angle", angle_summary.get("elbow_angle", {}).get("max")),
        "shoulder_angle_max":    get_angle_if_primary("shoulder_angle", angle_summary.get("shoulder_angle", {}).get("max")),
        "front_knee_angle_min":  get_angle_if_primary("front_knee_angle", angle_summary.get("front_knee_angle", {}).get("min")),
        "hip_shoulder_sep_max":  get_angle_if_primary("hip_shoulder_sep", angle_summary.get("hip_shoulder_sep", {}).get("max")),
        "trunk_lean_max":        get_angle_if_primary("trunk_lean", angle_summary.get("trunk_lean", {}).get("max")),
        "trunk_flexion_max":     get_angle_if_primary("trunk_flexion", angle_summary.get("trunk_flexion", {}).get("max")),
        "lateral_flexion_max":   get_angle_if_primary("lateral_flexion", angle_summary.get("lateral_flexion", {}).get("max")),
        "backfoot_angle_max":    get_angle_if_primary("backfoot_angle", angle_summary.get("backfoot_angle", {}).get("max")),
        "frontfoot_angle_max":   get_angle_if_primary("frontfoot_angle", angle_summary.get("frontfoot_angle", {}).get("max")),

        # Bowling Style
        "bowling_style":        bowling_style,
        "bowling_style_front_score": bowling_style_breakdown.get("front_score"),
        "bowling_style_side_score":  bowling_style_breakdown.get("side_score"),
        "bowling_style_mixed_score": bowling_style_breakdown.get("mixed_score"),

        # Run-up Metrics
        "runup_peak_momentum_frame":      runup_metrics.get("peak_momentum_frame"),
        "runup_backfoot_contact_frame":   runup_metrics.get("backfoot_contact_frame"),
        "runup_frontfoot_contact_frame":  runup_metrics.get("frontfoot_contact_frame"),
        "runup_approach_angle_deg":       runup_metrics.get("approach_angle_deg"),
        "runup_average_gate_width_px":    runup_metrics.get("average_gate_width_px"),
        "runup_stride_count":             runup_metrics.get("stride_count"),
        "runup_peak_velocity_px_frame":   runup_metrics.get("peak_velocity_px_frame"),
        "runup_momentum_at_backfoot":     runup_metrics.get("momentum_at_backfoot"),

        # Phase frame ranges
        # Note: phase_ranges from segment_phases has keys like "movement_start_frame", "jump_frame", etc.
        # We need to map these to phase names
        "phase_runup_start":         0,  # RUN-UP always starts at frame 0
        "phase_runup_end":           phase_ranges.get("movement_start_frame"),
        "phase_loadup_start":        phase_ranges.get("movement_start_frame"),
        "phase_loadup_end":          phase_ranges.get("jump_frame"),
        "phase_delivery_start":      phase_ranges.get("jump_frame"),
        "phase_delivery_end":        phase_ranges.get("peak_wrist_frame"),
        "phase_followthrough_start": phase_ranges.get("peak_wrist_frame"),
        "phase_followthrough_end":   None,  # Extends to end of video

        # Multi-view synchronization info
        "runup_view_used":           view_sync_info.get("runup_view_used"),
        "runup_view_fallback":       view_sync_info.get("runup_view_fallback"),
        "phase_view_used":           view_sync_info.get("phase_view_used"),
        "phase_view_fallback":       view_sync_info.get("front_view_fallback"),
        "sync_reference":            view_sync_info.get("sync_reference"),
        "sync_fallback_used":        view_sync_info.get("sync_fallback_used"),
        "relative_frames_load_up":   view_sync_info.get("relative_frames", {}).get("load_up"),
        "relative_frames_gather":    view_sync_info.get("relative_frames", {}).get("gather"),
        "relative_frames_delivery":  view_sync_info.get("relative_frames", {}).get("delivery"),
        "relative_frames_release":   view_sync_info.get("relative_frames", {}).get("release"),
        "relative_frames_follow_through": view_sync_info.get("relative_frames", {}).get("follow_through"),

        # ADD NEW FEATURES HERE
    }


def validate_record(record: dict) -> list:
    """
    Check for missing critical fields.

    Returns:
        List of warning strings. Empty = record is clean.
    """
    warnings = []
    critical = ["release_frame", "elbow_angle_at_release"]
    for field in critical:
        if record.get(field) is None:
            warnings.append(f"Missing critical field: {field}")
    return warnings