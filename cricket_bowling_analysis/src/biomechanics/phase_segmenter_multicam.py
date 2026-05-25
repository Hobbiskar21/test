"""
src/biomechanics/phase_segmenter_multicam.py
─────────────────────────────────────────────────────────────────────────────
Multi-camera phase detection using synchronized 3-camera setup.

CAMERA SETUP:
    - Front camera: bowler facing camera
    - Side camera: bowler from side (perpendicular)
    - Back camera: bowler from behind

PHASES (Multi-Camera):
    RUN-UP         : bowler approaching crease (all cameras)
    LOAD-GATHER    : TRIGGERED when side camera shows bowler in side-on position
                     (shoulders perpendicular to camera, ready to load)
    DELIVERY       : arm rotating → ball release (peak arm speed)
    FOLLOW-THROUGH : post-release forward motion

KEY INSIGHT:
    In a synchronized 3-camera setup, the side camera provides the clearest
    view of when the bowler transitions from run-up to load-gather.
    We detect this by monitoring shoulder orientation in the side camera.

APPROACH:
    1. Monitor side camera for shoulder perpendicularity (side-on position)
    2. When side camera shows clear side-on view → LOAD-GATHER starts
    3. Use other cameras for arm rotation and release detection
    4. Synchronize across all cameras using frame alignment
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Dict, List
import warnings


def _to_1d(data: ArrayLike, fill: float = 0.0) -> np.ndarray:
    """Convert list/array (possibly containing Nones) to float64 ndarray."""
    return np.array(
        [fill if v is None else float(v) for v in data],
        dtype=np.float64,
    )


def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average smoothing."""
    w = max(3, window)
    if len(signal) < w:
        return signal.copy()
    kernel = np.ones(w) / w
    padded = np.pad(signal, w // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(signal)]


def _smooth_labels(labels: list, window: int = 7) -> list:
    """Stabilise phase label sequence using sliding window majority vote."""
    n = len(labels)
    result = labels.copy()
    half = window // 2

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = labels[lo:hi]

        counts = {}
        for lbl in chunk:
            counts[lbl] = counts.get(lbl, 0) + 1
        result[i] = max(counts, key=counts.get)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SIDE CAMERA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def detect_side_on_position(
    side_camera_landmarks: List,
    frame_width: int,
    frame_height: int,
    fps: float,
) -> int:
    """
    Detect when side camera shows bowler in clear side-on position.
    This marks the transition from RUN-UP to LOAD-GATHER.

    Algorithm:
        1. Extract shoulder positions from side camera
        2. Calculate shoulder width (distance between left and right shoulder)
        3. In side-on view, shoulders should be narrow (perpendicular to camera)
        4. Find frame where shoulder width is minimum (most perpendicular)
        5. This is when bowler is in side-on position ready to load

    Parameters
    ----------
    side_camera_landmarks : list
        List of landmark arrays from side camera (17 COCO keypoints per frame)
    frame_width : int
        Frame width in pixels
    frame_height : int
        Frame height in pixels
    fps : float
        Frames per second

    Returns
    -------
    int
        Frame index where side-on position is detected
    """
    if not side_camera_landmarks or len(side_camera_landmarks) < 10:
        return max(1, int(len(side_camera_landmarks) * 0.15))

    n = len(side_camera_landmarks)

    # COCO indices: 5=left_shoulder, 6=right_shoulder
    L_SHOULDER = 5
    R_SHOULDER = 6

    shoulder_widths = []

    for frame_idx, landmarks in enumerate(side_camera_landmarks):
        if landmarks is None or len(landmarks) < 7:
            shoulder_widths.append(frame_width)  # Invalid frame
            continue

        l_shoulder = landmarks[L_SHOULDER]
        r_shoulder = landmarks[R_SHOULDER]

        # Check visibility
        if l_shoulder.visibility < 0.5 or r_shoulder.visibility < 0.5:
            shoulder_widths.append(frame_width)  # Invalid frame
            continue

        # Calculate shoulder width in pixels
        shoulder_width = abs(r_shoulder.x - l_shoulder.x) * frame_width
        shoulder_widths.append(shoulder_width)

    shoulder_widths = np.array(shoulder_widths)

    # Smooth to remove noise
    smooth_w = max(3, int(fps * 0.1))
    shoulder_smooth = _smooth_signal(shoulder_widths, smooth_w)

    # Find minimum shoulder width (most perpendicular = side-on position)
    # Search in first 50% of video (before delivery)
    search_region = shoulder_smooth[: int(n * 0.5)]
    if len(search_region) == 0:
        return max(1, int(n * 0.15))

    min_idx = int(np.argmin(search_region))
    side_on_frame = min_idx

    print(f"[PHASE-MULTICAM] Side-on position detected at frame {side_on_frame}")
    print(f"[PHASE-MULTICAM] Shoulder width: {shoulder_widths[side_on_frame]:.1f}px")

    return side_on_frame


def detect_jump_peak_multicam(
    ankle_positions: List,
    fps: float,
) -> int:
    """
    Detect jump peak from any camera (typically front or back).
    This marks the transition from RUN-UP to LOAD-GATHER (fallback).

    Parameters
    ----------
    ankle_positions : list
        List of [(left_ankle, right_ankle)] tuples per frame
    fps : float
        Frames per second

    Returns
    -------
    int
        Frame index of jump peak
    """
    if not ankle_positions or len(ankle_positions) < 10:
        return max(1, int(len(ankle_positions) * 0.15))

    n = len(ankle_positions)

    # Extract Y for each ankle
    left_y = []
    right_y = []

    for frame_ankles in ankle_positions:
        if frame_ankles is None or len(frame_ankles) < 2:
            left_y.append(0.5)
            right_y.append(0.5)
            continue

        l_a = frame_ankles[0]
        r_a = frame_ankles[1]
        left_y.append(float(l_a[1]) if l_a is not None else 0.5)
        right_y.append(float(r_a[1]) if r_a is not None else 0.5)

    left_y = np.array(left_y)
    right_y = np.array(right_y)

    # Smooth
    w_smooth = max(3, int(fps * 0.15))
    left_smooth = _smooth_signal(left_y, w_smooth)
    right_smooth = _smooth_signal(right_y, w_smooth)

    # Vertical velocity
    left_vel = np.gradient(left_smooth)
    right_vel = np.gradient(right_smooth)

    # Find frames where BOTH ankles move upward
    neg_l = left_vel[left_vel < 0]
    neg_r = right_vel[right_vel < 0]
    l_thr = np.percentile(neg_l, 50) if len(neg_l) > 0 else -0.01
    r_thr = np.percentile(neg_r, 50) if len(neg_r) > 0 else -0.01

    airborne_frames = [
        i for i in range(5, n)
        if left_vel[i] < l_thr and right_vel[i] < r_thr
    ]

    if not airborne_frames:
        return max(1, int(n * 0.15))

    # Group into intervals
    intervals = []
    current = [airborne_frames[0]]

    for i in range(1, len(airborne_frames)):
        if airborne_frames[i] - airborne_frames[i - 1] <= 2:
            current.append(airborne_frames[i])
        else:
            if len(current) >= 3:
                intervals.append(current)
            current = [airborne_frames[i]]

    if len(current) >= 3:
        intervals.append(current)

    if not intervals:
        intervals = [airborne_frames]

    # Pick interval with lowest combined Y
    best_interval = min(
        intervals,
        key=lambda iv: min(left_smooth[f] + right_smooth[f] for f in iv)
    )

    jump_frame = min(
        best_interval,
        key=lambda f: left_smooth[f] + right_smooth[f]
    )

    print(f"[PHASE-MULTICAM] Jump peak at frame {jump_frame}")
    return int(jump_frame)


def detect_arm_rotation_start_multicam(
    elbow_angles: List,
    fps: float,
) -> Optional[int]:
    """
    Detect start of arm rotation from any camera.
    This marks the transition from LOAD-GATHER to DELIVERY.

    Parameters
    ----------
    elbow_angles : list
        List of elbow angles per frame
    fps : float
        Frames per second

    Returns
    -------
    int or None
        Frame index or None if not detected
    """
    if not elbow_angles or len(elbow_angles) < 10:
        return None

    elbow_arr = _to_1d(elbow_angles, fill=0.0)
    smooth_w = max(3, int(fps * 0.05))
    elbow_smooth = _smooth_signal(elbow_arr, smooth_w)

    # Angular velocity
    angular_vel = np.gradient(elbow_smooth)

    # Find where arm is straightening fastest
    search_start = int(len(angular_vel) * 0.25)
    search_region = angular_vel[search_start:]

    if len(search_region) == 0:
        return None

    min_idx = int(np.argmin(search_region))
    arm_rotation_frame = search_start + min_idx

    print(f"[PHASE-MULTICAM] Arm rotation start at frame {arm_rotation_frame}")
    return arm_rotation_frame


def detect_ball_release_multicam(
    wrist_velocities: List,
    elbow_angles: List,
    fps: float,
) -> int:
    """
    Detect ball release from any camera.
    This marks the transition from DELIVERY to FOLLOW-THROUGH.

    Parameters
    ----------
    wrist_velocities : list
        List of wrist velocities per frame
    elbow_angles : list
        List of elbow angles per frame
    fps : float
        Frames per second

    Returns
    -------
    int
        Frame index of ball release
    """
    if not wrist_velocities or len(wrist_velocities) < 10:
        return max(1, int(len(wrist_velocities) * 0.6))

    wrist_v_arr = _to_1d(wrist_velocities, fill=0.0)
    elbow_arr = _to_1d(elbow_angles, fill=0.0)

    smooth_w = max(3, int(fps * 0.05))
    wrist_smooth = _smooth_signal(wrist_v_arr, smooth_w)

    # Find frame with maximum wrist velocity
    release_frame = int(np.argmax(wrist_smooth))

    if release_frame < len(elbow_arr):
        elbow_angle = elbow_arr[release_frame]
        print(f"[PHASE-MULTICAM] Ball release at frame {release_frame} (elbow: {elbow_angle:.1f}°)")
    else:
        print(f"[PHASE-MULTICAM] Ball release at frame {release_frame}")

    return release_frame


# ─────────────────────────────────────────────────────────────────────────────
# PHASE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_phases_multicam(
    n_frames: int,
    side_on_frame: int,
    arm_rotation_start: Optional[int],
    release_frame: int,
    smooth_window: int = 7,
) -> list:
    """
    Assign phase label to every frame using multi-camera events.

    Boundaries:
        [0,              side_on_frame)      → RUN-UP
        [side_on_frame,  arm_rotation_start) → LOAD-GATHER
        [arm_rotation,   release_frame]      → DELIVERY
        [release_frame+1, end]               → FOLLOW-THROUGH

    Parameters
    ----------
    n_frames : int
        Total number of frames
    side_on_frame : int
        Frame where side camera shows side-on position
    arm_rotation_start : int or None
        Frame where arm rotation starts
    release_frame : int
        Frame where ball is released
    smooth_window : int
        Window size for smoothing

    Returns
    -------
    list
        Phase labels for each frame
    """
    # Use arm rotation start if found, else midpoint
    if arm_rotation_start is None:
        arm_rotation_start = side_on_frame + max(1, (release_frame - side_on_frame) // 2)

    # Clamp boundaries
    side_on_frame = max(1, min(side_on_frame, n_frames - 3))
    arm_rotation_start = max(side_on_frame + 1, min(arm_rotation_start, n_frames - 2))
    release_frame = max(arm_rotation_start + 1, min(release_frame, n_frames - 1))

    # Initial assignment
    labels = ["RUN-UP"] * n_frames

    for i in range(side_on_frame, n_frames):
        labels[i] = "LOAD-GATHER"

    for i in range(arm_rotation_start, n_frames):
        labels[i] = "DELIVERY"

    for i in range(release_frame + 1, n_frames):
        labels[i] = "FOLLOW-THROUGH"

    # Stabilise
    labels = _smooth_labels(labels, window=smooth_window)

    return labels


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def segment_phases_multicam(
    side_camera_landmarks: List,
    wrist_velocities: List,
    elbow_angles: List,
    frame_width: int,
    frame_height: int,
    fps: float,
    ankle_positions: List = None,
) -> tuple[np.ndarray, dict]:
    """
    Multi-camera phase segmentation using synchronized 3-camera setup.

    LOAD-GATHER phase is triggered when side camera shows bowler in side-on position.

    Parameters
    ----------
    side_camera_landmarks : list
        Landmarks from side camera (17 COCO keypoints per frame)
    wrist_velocities : list
        Wrist velocities per frame
    elbow_angles : list
        Elbow angles per frame
    frame_width : int
        Frame width in pixels
    frame_height : int
        Frame height in pixels
    fps : float
        Frames per second
    ankle_positions : list, optional
        Ankle positions for jump detection (fallback)

    Returns
    -------
    labels : np.ndarray
        Phase labels for each frame
    boundaries : dict
        Dictionary with frame indices for each phase boundary
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")

    n = len(side_camera_landmarks)
    if n == 0:
        raise ValueError("side_camera_landmarks is empty.")

    print(f"[PHASE-MULTICAM] Segmenting {n} frames at {fps} fps (3-camera mode)")

    # ── Event detection ───────────────────────────────────────────────────

    # 1. Side-on position (RUN-UP → LOAD-GATHER boundary)
    # PRIMARY: Use side camera to detect when bowler is in side-on position
    side_on_frame = detect_side_on_position(
        side_camera_landmarks, frame_width, frame_height, fps
    )

    # 2. Arm rotation start (LOAD-GATHER → DELIVERY boundary)
    if elbow_angles is not None and len(elbow_angles) > 0:
        arm_rotation_start = detect_arm_rotation_start_multicam(elbow_angles, fps)
    else:
        arm_rotation_start = None
        print(f"[PHASE-MULTICAM] Arm rotation start not detected")

    # 3. Ball release (DELIVERY → FOLLOW-THROUGH boundary)
    if wrist_velocities is not None and len(wrist_velocities) > 0:
        release_frame = detect_ball_release_multicam(wrist_velocities, elbow_angles or [], fps)
    else:
        release_frame = max(1, int(n * 0.6))
        print(f"[PHASE-MULTICAM] Ball release fallback at frame {release_frame}")

    # ── Phase classification ──────────────────────────────────────────────
    labels_list = classify_phases_multicam(
        n_frames=n,
        side_on_frame=side_on_frame,
        arm_rotation_start=arm_rotation_start,
        release_frame=release_frame,
        smooth_window=7,
    )

    labels = np.array(labels_list, dtype=object)

    # ── Summary ───────────────────────────────────────────────────────────
    counts = {
        p: int(np.sum(labels == p))
        for p in ["RUN-UP", "LOAD-GATHER", "DELIVERY", "FOLLOW-THROUGH"]
    }
    print(
        f"[PHASE-MULTICAM] DONE | "
        f"RUN-UP:{counts['RUN-UP']}f "
        f"LOAD:{counts['LOAD-GATHER']}f "
        f"DEL:{counts['DELIVERY']}f "
        f"FT:{counts['FOLLOW-THROUGH']}f"
    )

    boundaries = {
        "side_on_frame": side_on_frame,
        "arm_rotation_start_frame": arm_rotation_start,
        "release_frame": release_frame,
        "movement_start_frame": side_on_frame,
        "peak_wrist_frame": release_frame,
    }

    return labels, boundaries
