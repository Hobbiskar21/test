"""
src/biomechanics/release_detector.py
--------------------------------------
Finds the exact frame the ball is released.

Combines two signals:
    1. Wrist velocity peak (always used)
    2. Wrist-to-ball distance spike (used if ball tracking available)

Searches only within the DELIVERY phase window.
"""

import numpy as np
from typing import Optional


def find_release_frame(wrist_velocity_seq: list,
                       ball_positions: list = None,
                       wrist_positions: list = None,
                       phase_map: dict = None) -> Optional[int]:
    """
    Find the release frame index.

    Args:
        wrist_velocity_seq : list of float or None per frame.
        ball_positions     : list of (x, y) or None per frame (from DeepSORT).
        wrist_positions    : list of (x, y) or None per frame.
        phase_map          : dict frame_idx -> phase. Restricts search to DELIVERY.

    Returns:
        Frame index (int) or None if not detected.
    """
    n = len(wrist_velocity_seq)

    if phase_map is not None:
        search_frames = [i for i in range(n) if phase_map.get(i) == "DELIVERY"]
        if not search_frames:
            print(f"[RELEASE] No DELIVERY phase frames found. Phase map has {len(phase_map)} entries")
            print(f"[RELEASE] Phases in map: {set(phase_map.values())}")
            # Fallback: search entire sequence
            search_frames = list(range(n))
    else:
        search_frames = list(range(n))

    if not search_frames:
        print(f"[RELEASE] No frames to search")
        return None

    # Signal 1: wrist velocity
    wrist_score = np.zeros(n)
    valid_velocities = 0
    for i in search_frames:
        v = wrist_velocity_seq[i]
        if v is not None and not np.isnan(v) and not np.isinf(v):
            wrist_score[i] = v
            valid_velocities += 1

    if valid_velocities == 0:
        print(f"[RELEASE] No valid wrist velocities found")
        return None

    # Signal 2: wrist-to-ball distance
    ball_score = np.zeros(n)
    if ball_positions is not None and wrist_positions is not None:
        for i in search_frames:
            bp = ball_positions[i]
            wp = wrist_positions[i]
            if bp is not None and wp is not None:
                ball_score[i] = np.linalg.norm(
                    np.array(bp) - np.array(wp)
                )

    # Normalize and combine
    def normalize(arr):
        mx = arr.max()
        return arr / mx if mx > 0 else arr

    combined = normalize(wrist_score)
    if ball_positions is not None:
        combined += normalize(ball_score)

    # Mask to search window
    mask = np.zeros(n)
    for i in search_frames:
        mask[i] = 1
    combined *= mask

    release_frame = int(np.argmax(combined))
    
    if combined[release_frame] > 0:
        print(f"[RELEASE] Release frame detected at {release_frame} (score: {combined[release_frame]:.2f})")
        return release_frame
    else:
        print(f"[RELEASE] No release frame detected (max score: {combined[release_frame]:.2f})")
        return None