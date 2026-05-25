"""
src/visualization/feature_priority_display.py
──────────────────────────────────────────────
Display biomechanical features with color-coded priority levels.

Color scheme:
- GREEN (0, 255, 0): Primary view (confidence 1.0)
- YELLOW (0, 255, 255): Secondary view (confidence 0.5-0.7)
- RED (0, 0, 255): Fallback/low confidence
- GRAY (128, 128, 128): Not available
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
from src.biomechanics.view_priority_framework import (
    get_feature_priority, get_confidence_score, ViewType
)


# Color mapping for confidence levels (BGR format)
CONFIDENCE_COLORS = {
    1.0: (0, 255, 0),      # GREEN - Primary view (100% confidence)
    0.7: (0, 200, 0),      # Dark green - High secondary
    0.65: (0, 180, 0),     # Dark green - Medium secondary
    0.6: (0, 160, 0),      # Dark green - Low secondary
    0.55: (0, 140, 0),     # Darker green
    0.5: (0, 255, 255),    # YELLOW - Fallback
    0.45: (0, 200, 200),   # Dark yellow
    0.4: (0, 150, 150),    # Darker yellow
    0.0: (128, 128, 128),  # GRAY - Not available
}


def get_color_for_confidence(confidence: float) -> Tuple[int, int, int]:
    """Get BGR color for a confidence score."""
    # Find closest confidence level
    closest = min(CONFIDENCE_COLORS.keys(), key=lambda x: abs(x - confidence))
    return CONFIDENCE_COLORS[closest]


def draw_feature_priority_box(
    img: np.ndarray,
    features: Dict[str, Optional[float]],
    angle_metadata: Dict[str, Dict],
    position: Tuple[int, int] = (10, 100),
    box_width: int = 350,
    line_height: int = 20,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw a box showing all features with color-coded priority levels.
    
    Parameters
    ----------
    img : np.ndarray
        Image to draw on
    features : dict
        {feature_name: value} - computed features
    angle_metadata : dict
        {feature_name: {view, confidence, ...}} - metadata for each feature
    position : tuple
        (x, y) top-left corner of box
    box_width : int
        Width of the feature box
    line_height : int
        Height of each feature line
    font_scale : float
        Font scale for text
    
    Returns
    -------
    np.ndarray
        Image with feature box drawn
    """
    out = img.copy()
    h, w = img.shape[:2]
    
    x, y = position
    
    # Sort features by priority (primary first)
    sorted_features = []
    for fname, fvalue in sorted(features.items()):
        priority = get_feature_priority(fname)
        if priority:
            sorted_features.append((fname, fvalue, priority))
    
    # Calculate box height
    num_features = len(sorted_features)
    box_height = (num_features + 1) * line_height + 10
    
    # Ensure box fits within frame - adjust position if needed
    if y + box_height > h - 20:  # Leave 20px margin at bottom
        y = max(10, h - box_height - 20)
    
    # Draw semi-transparent background box
    overlay = out.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)
    
    # Draw box border
    cv2.rectangle(out, (x, y), (x + box_width, y + box_height), (255, 255, 255), 2)
    
    # Draw title
    cv2.putText(
        out, "FEATURE PRIORITY", (x + 10, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale + 0.1, (255, 255, 255), 1, cv2.LINE_AA
    )
    
    # Draw each feature
    for i, (fname, fvalue, priority) in enumerate(sorted_features):
        line_y = y + 25 + i * line_height
        
        # Get metadata
        metadata = angle_metadata.get(fname, {})
        confidence = metadata.get("confidence", 0.0)
        view_used = metadata.get("view", "unknown")
        is_primary = metadata.get("is_primary_view", False)
        
        # Get color based on confidence
        color = get_color_for_confidence(confidence)
        
        # Draw colored indicator circle
        cv2.circle(out, (x + 15, line_y - 5), 5, color, -1)
        
        # Draw feature name
        display_name = fname.replace("_", " ").title()
        cv2.putText(
            out, display_name, (x + 30, line_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )
        
        # Draw value and confidence
        if fvalue is not None:
            # Format: "123.4deg (100%)"
            confidence_pct = int(confidence * 100)
            value_str = f"{fvalue:.1f}deg ({confidence_pct}%)"
        else:
            value_str = "N/A"
        
        cv2.putText(
            out, value_str, (x + 200, line_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.1, color, 1, cv2.LINE_AA
        )
        
        # Draw view indicator
        view_indicator = "P" if is_primary else "S"  # P=Primary, S=Secondary
        cv2.putText(
            out, f"[{view_indicator}]", (x + 320, line_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.1, color, 1, cv2.LINE_AA
        )
    
    # Draw legend at bottom
    legend_y = y + box_height + 20
    cv2.putText(
        out, "P=Primary  S=Secondary  Green=100%  Yellow=Fallback",
        (x, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale - 0.1, (200, 200, 200), 1, cv2.LINE_AA
    )
    
    return out


def draw_view_status_bar(
    img: np.ndarray,
    view_used: str,
    view_fallback: bool,
    position: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """
    Draw a status bar showing which view is being used.
    
    Parameters
    ----------
    img : np.ndarray
        Image to draw on
    view_used : str
        View name ("side", "front", "back")
    view_fallback : bool
        Whether this is a fallback view
    position : tuple
        (x, y) position for the status bar
    
    Returns
    -------
    np.ndarray
        Image with status bar drawn
    """
    out = img.copy()
    x, y = position
    
    # Color based on fallback status
    color = (0, 255, 0) if not view_fallback else (0, 255, 255)  # Green or Yellow
    
    # Draw status text
    status_text = f"View: {view_used.upper()}"
    if view_fallback:
        status_text += " (FALLBACK)"
    
    cv2.putText(
        out, status_text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
    )
    
    return out


def draw_sync_info(
    img: np.ndarray,
    sync_info: Dict,
    position: Tuple[int, int] = (10, 60),
) -> np.ndarray:
    """
    Draw multi-view synchronization information.
    
    Parameters
    ----------
    img : np.ndarray
        Image to draw on
    sync_info : dict
        Sync information from view_priority_sync
    position : tuple
        (x, y) position for the sync info
    
    Returns
    -------
    np.ndarray
        Image with sync info drawn
    """
    out = img.copy()
    x, y = position
    
    sync_ref = sync_info.get("sync_reference", "unknown")
    sync_fallback = sync_info.get("sync_fallback_used", False)
    
    color = (0, 255, 0) if not sync_fallback else (0, 255, 255)
    
    sync_text = f"Sync: {sync_ref.upper()}"
    if sync_fallback:
        sync_text += " (FALLBACK)"
    
    cv2.putText(
        out, sync_text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA
    )
    
    return out
