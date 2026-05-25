"""
src/biomechanics/velocity_estimator.py
----------------------------------------
Estimates joint velocities via numerical differentiation.
v[t] = (position[t+1] - position[t-1]) / (2 * dt)

Computes:
    arm_velocity_max/mean  : bowling wrist speed
    runup_speed_mean       : hip midpoint speed during run-up phase
"""

import numpy as np
from typing import Optional


def compute_velocity_sequence(positions: list, fps: float) -> list:
    """
    Frame-by-frame speed from position sequence using central differences.

    Args:
        positions : list of (x, y) arrays or None per frame.
        fps       : video frame rate.

    Returns:
        List of float speeds or None — same length as positions.
    """
    dt = 1.0 / fps
    n  = len(positions)
    velocities = [None] * n

    for i in range(1, n - 1):
        prev = positions[i - 1]
        nxt  = positions[i + 1]
        if prev is None or nxt is None:
            continue
        displacement = np.array(nxt) - np.array(prev)
        velocities[i] = round(float(np.linalg.norm(displacement) / (2.0 * dt)), 4)

    return velocities


def extract_joint_positions(landmarks_sequence: list, joint_index: int,
                             width: int, height: int,
                             min_confidence: float = 0.5) -> list:
    """Extract pixel-space (x, y) for one joint across all frames."""
    positions = []
    for landmarks in landmarks_sequence:
        if landmarks is None:
            positions.append(None)
            continue
        lm = landmarks[joint_index]
        if lm.visibility < min_confidence:
            positions.append(None)
        else:
            positions.append(np.array([lm.x * width, lm.y * height]))
    return positions


def peak_velocity(seq: list) -> Optional[float]:
    valid = [v for v in seq if v is not None]
    return round(float(max(valid)), 4) if valid else None


def mean_velocity(seq: list) -> Optional[float]:
    valid = [v for v in seq if v is not None]
    return round(float(np.mean(valid)), 4) if valid else None


def compute_all_velocities(landmarks_sequence: list, fps: float,
                            width: int, height: int,
                            phase_map: dict = None) -> dict:
    """
    Compute all velocity metrics from a full delivery landmark sequence.
    
    Uses COCO keypoint indices:
    - 10 = right wrist
    - 11 = left hip
    - 12 = right hip

    Returns:
        dict with arm_velocity_max, arm_velocity_mean, runup_speed_mean,
        wrist_velocity_seq, hip_velocity_seq (full sequences for storage).
    """
    # Bowling wrist — right wrist = COCO index 10
    wrist_positions  = extract_joint_positions(landmarks_sequence, 10, width, height)
    wrist_velocities = compute_velocity_sequence(wrist_positions, fps)

    # Hip midpoint — average of left (11) and right (12) hips (COCO indices)
    l_hip = extract_joint_positions(landmarks_sequence, 11, width, height)
    r_hip = extract_joint_positions(landmarks_sequence, 12, width, height)

    mid_hip = []
    for lh, rh in zip(l_hip, r_hip):
        if lh is not None and rh is not None:
            mid_hip.append((np.array(lh) + np.array(rh)) / 2.0)
        else:
            mid_hip.append(None)

    hip_velocities = compute_velocity_sequence(mid_hip, fps)

    # Run-up speed: mean hip velocity during RUN-UP phase only
    if phase_map is not None:
        runup_vels = [v for i, v in enumerate(hip_velocities)
                      if phase_map.get(i) == "RUN-UP" and v is not None]
        runup_speed = round(float(np.mean(runup_vels)), 4) if runup_vels else None
    else:
        runup_speed = mean_velocity(hip_velocities)

    return {
        "arm_velocity_max":   peak_velocity(wrist_velocities),
        "arm_velocity_mean":  mean_velocity(wrist_velocities),
        "runup_speed_mean":   runup_speed,
        "wrist_velocity_seq": wrist_velocities,
        "hip_velocity_seq":   hip_velocities,
    }