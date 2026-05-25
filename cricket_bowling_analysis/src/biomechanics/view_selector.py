"""
src/biomechanics/view_selector.py
----------------------------------
View selection logic for multi-camera analysis.

Enforces strict view prioritization:
- RUN-UP: BACK → FRONT → SIDE (use only one)
- LOAD-UP/GATHER/DELIVERY/FOLLOW-THROUGH: FRONT → SIDE (never BACK)

Adds metadata for view usage tracking.
"""

from typing import Dict, Optional, List


class ViewSelector:
    """
    Select appropriate camera view for each phase.
    
    Rules:
    - RUN-UP: Prefer BACK (shows approach), fallback to FRONT, then SIDE
    - LOAD-UP/GATHER/DELIVERY/FOLLOW-THROUGH: Prefer FRONT (shows arm action), fallback to SIDE
    - Never use BACK for delivery phases
    """
    
    PHASE_RUNUP = "RUN-UP"
    PHASE_LOAD = "LOAD-GATHER"
    PHASE_DELIVERY = "DELIVERY"
    PHASE_FOLLOWTHROUGH = "FOLLOW-THROUGH"
    
    def __init__(self):
        """Initialize view selector."""
        self.view_usage_log = []
    
    def select_view_for_phase(self, phase: str, available_views: Dict[str, bool]) -> Dict:
        """
        Select best view for a given phase.
        
        Parameters
        ----------
        phase : str
            Phase name: "RUN-UP", "LOAD-GATHER", "DELIVERY", "FOLLOW-THROUGH"
        available_views : dict
            {"back": bool, "front": bool, "side": bool}
        
        Returns
        -------
        dict
            {
                "view_used": "back/front/side",
                "view_fallback": bool,
                "available_views": list of available views,
                "phase": phase name
            }
        """
        if phase == self.PHASE_RUNUP:
            return self._select_runup_view(available_views)
        else:
            # LOAD-GATHER, DELIVERY, FOLLOW-THROUGH
            return self._select_delivery_phase_view(available_views, phase)
    
    def _select_runup_view(self, available_views: Dict[str, bool]) -> Dict:
        """
        Select view for RUN-UP phase.
        Priority: BACK → FRONT → SIDE
        """
        if available_views.get("back", False):
            return {
                "view_used": "back",
                "view_fallback": False,
                "available_views": [v for v, avail in available_views.items() if avail],
                "phase": self.PHASE_RUNUP,
            }
        elif available_views.get("front", False):
            return {
                "view_used": "front",
                "view_fallback": True,
                "available_views": [v for v, avail in available_views.items() if avail],
                "phase": self.PHASE_RUNUP,
            }
        elif available_views.get("side", False):
            return {
                "view_used": "side",
                "view_fallback": True,
                "available_views": [v for v, avail in available_views.items() if avail],
                "phase": self.PHASE_RUNUP,
            }
        else:
            return {
                "view_used": None,
                "view_fallback": True,
                "available_views": [],
                "phase": self.PHASE_RUNUP,
                "error": "No views available",
            }
    
    def _select_delivery_phase_view(self, available_views: Dict[str, bool], phase: str) -> Dict:
        """
        Select view for LOAD-GATHER, DELIVERY, FOLLOW-THROUGH phases.
        Priority: FRONT → SIDE (NEVER BACK)
        """
        if available_views.get("front", False):
            return {
                "view_used": "front",
                "view_fallback": False,
                "available_views": [v for v, avail in available_views.items() if avail],
                "phase": phase,
            }
        elif available_views.get("side", False):
            return {
                "view_used": "side",
                "view_fallback": True,
                "available_views": [v for v, avail in available_views.items() if avail],
                "phase": phase,
            }
        else:
            return {
                "view_used": None,
                "view_fallback": True,
                "available_views": [v for v, avail in available_views.items() if avail],
                "phase": phase,
                "error": "No suitable views available (BACK not allowed for delivery phases)",
            }
    
    def get_view_metadata(self, phase_map: Dict[int, str], 
                         available_views: Dict[str, bool]) -> Dict:
        """
        Generate view metadata for entire delivery.
        
        Parameters
        ----------
        phase_map : dict
            {frame_idx: phase_name}
        available_views : dict
            {"back": bool, "front": bool, "side": bool}
        
        Returns
        -------
        dict
            View usage metadata for all phases
        """
        metadata = {
            "view_selection_applied": True,
            "available_views": [v for v, avail in available_views.items() if avail],
            "phase_views": {},
        }
        
        # Get unique phases
        unique_phases = set(phase_map.values())
        
        for phase in unique_phases:
            view_info = self.select_view_for_phase(phase, available_views)
            metadata["phase_views"][phase] = view_info
        
        return metadata
