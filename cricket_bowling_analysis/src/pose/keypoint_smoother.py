"""
src/pose/keypoint_smoother.py
-------------------------------
Apply the same rolling-window smoothing logic used by src/scripts/metrics_util.py
and src/scripts/event_detect.py, but on person keypoints for the main pipeline.
"""

import numpy as np
import pandas as pd

from .keypoint_converter import SmoothedLandmark
from src.utils.config_loader import get_config


def _smooth_sequence(values: list, window: int = 5) -> list:
    """
    Mirror src/scripts/metrics_util.smooth():
    - centered rolling mean
    - min_periods=1
    - bfill then ffill
    """
    if not values:
        return []

    series = pd.Series([np.nan if value is None else value for value in values], dtype=float)
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
    smoothed = smoothed.bfill().ffill()
    return smoothed.tolist()


def smooth_coco_keypoints_sequence(kpts_sequence: list, window: int | None = None) -> list:
    """
    Apply the script-style rolling smoother directly to COCO keypoints in pixel space.

    Parameters
    ----------
    kpts_sequence : list
        List of (17, 2) numpy arrays or None.
    window : int, optional
        Rolling window size. Defaults to config pose.smoothing_window.

    Returns
    -------
    list
        List of smoothed (17, 2) numpy arrays or None.
    """
    cfg = get_config()
    if window is None:
        window = int(cfg.get("pose", {}).get("smoothing_window", 5))
    window = max(1, window)

    if not kpts_sequence:
        return []

    xs = [[] for _ in range(17)]
    ys = [[] for _ in range(17)]

    for kpts in kpts_sequence:
        for j in range(17):
            if kpts is not None and j < len(kpts):
                xs[j].append(float(kpts[j][0]))
                ys[j].append(float(kpts[j][1]))
            else:
                xs[j].append(None)
                ys[j].append(None)

    smoothed_xs = [_smooth_sequence(xs[j], window) for j in range(17)]
    smoothed_ys = [_smooth_sequence(ys[j], window) for j in range(17)]

    result = []
    for i, kpts in enumerate(kpts_sequence):
        if kpts is None:
            result.append(None)
            continue

        smoothed = np.array(kpts, dtype=np.float32, copy=True)
        for j in range(min(17, len(smoothed))):
            x_value = smoothed_xs[j][i]
            y_value = smoothed_ys[j][i]
            if x_value is not None and not np.isnan(x_value):
                smoothed[j, 0] = x_value
            if y_value is not None and not np.isnan(y_value):
                smoothed[j, 1] = y_value
        result.append(smoothed)

    return result


def smooth_landmarks_sequence(landmarks_sequence: list) -> list:
    """
    Smooth x and y of all 17 COCO landmarks using the script rolling smoother.

    Returns a sequence matching the input length and preserves None frames so
    the rest of the pipeline behavior stays unchanged.
    """
    cfg = get_config()
    window = int(cfg.get("pose", {}).get("smoothing_window", 5))
    window = max(1, window)
    n = len(landmarks_sequence)

    xs = [[] for _ in range(17)]
    ys = [[] for _ in range(17)]

    for lm_list in landmarks_sequence:
        for j in range(17):
            if lm_list is not None and j < len(lm_list):
                xs[j].append(lm_list[j].x)
                ys[j].append(lm_list[j].y)
            else:
                xs[j].append(None)
                ys[j].append(None)

    smoothed_xs = [_smooth_sequence(xs[j], window) for j in range(17)]
    smoothed_ys = [_smooth_sequence(ys[j], window) for j in range(17)]

    result = []
    for i in range(n):
        if landmarks_sequence[i] is None:
            result.append(None)
            continue

        frame_landmarks = []
        for j in range(17):
            if j < len(landmarks_sequence[i]):
                original = landmarks_sequence[i][j]
                x_value = smoothed_xs[j][i]
                y_value = smoothed_ys[j][i]
                frame_landmarks.append(
                    SmoothedLandmark(
                        x=original.x if x_value is None or np.isnan(x_value) else x_value,
                        y=original.y if y_value is None or np.isnan(y_value) else y_value,
                        z=original.z,
                        visibility=original.visibility,
                    )
                )
            else:
                frame_landmarks.append(SmoothedLandmark(0, 0, 0, 0))
        result.append(frame_landmarks)

    return result
