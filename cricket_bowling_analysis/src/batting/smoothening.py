"""Batting-specific smoothing helpers."""

import numpy as np
import pandas as pd


def _smooth_coordinate(values: list, median_window: int = 5, mean_window: int = 7) -> list:
    """Smooth one x/y coordinate track.

    This is deliberately more stable than the previous exponential smoother:
    interpolation fills short missing gaps, the rolling median suppresses
    one-frame pose jumps, and the rolling mean gives a stable batting overlay.
    """
    if not values:
        return []

    series = pd.Series([np.nan if value is None else value for value in values], dtype=float)
    series = series.interpolate(limit_direction="both").bfill().ffill()
    denoised = series.rolling(window=median_window, center=True, min_periods=1).median()
    smoothed = denoised.rolling(window=mean_window, center=True, min_periods=1).mean()
    return smoothed.bfill().ffill().tolist()


def median_mean_smooth_coco_keypoints_sequence(
    kpts_sequence: list,
    median_window: int = 5,
    mean_window: int = 7,
) -> list:
    """Apply batting-specific median + mean smoothing to COCO keypoints.

    Parameters
    ----------
    kpts_sequence : list
        List of frame keypoints, each either None or an array-like of shape (17, 2).
    median_window : int
        Centered rolling median window used to remove detector spikes.
    mean_window : int
        Centered rolling mean window used to smooth normal motion.

    Returns
    -------
    list
        Smoothed COCO keypoints sequence with the same length as the input.
    """
    median_window = max(1, int(median_window))
    mean_window = max(1, int(mean_window))
    if not kpts_sequence:
        return []

    xs = [[] for _ in range(17)]
    ys = [[] for _ in range(17)]
    valid_frame = []

    for keypoints in kpts_sequence:
        valid_frame.append(keypoints is not None)
        for idx in range(17):
            if keypoints is not None and idx < len(keypoints):
                try:
                    x = float(keypoints[idx][0])
                    y = float(keypoints[idx][1])
                except (TypeError, ValueError, IndexError):
                    x, y = np.nan, np.nan
                xs[idx].append(None if np.isnan(x) else x)
                ys[idx].append(None if np.isnan(y) else y)
            else:
                xs[idx].append(None)
                ys[idx].append(None)

    smooth_xs = [_smooth_coordinate(values, median_window, mean_window) for values in xs]
    smooth_ys = [_smooth_coordinate(values, median_window, mean_window) for values in ys]

    smoothed_sequence = []
    for frame_idx, original in enumerate(kpts_sequence):
        if original is None:
            smoothed_sequence.append(None)
            continue
        smoothed_frame = np.array(original, dtype=np.float32, copy=True)
        for idx in range(min(17, len(smoothed_frame))):
            smoothed_frame[idx, 0] = smooth_xs[idx][frame_idx]
            smoothed_frame[idx, 1] = smooth_ys[idx][frame_idx]
        smoothed_sequence.append(smoothed_frame)

    return smoothed_sequence


def exponential_smooth_coco_keypoints_sequence(kpts_sequence: list, alpha: float = 0.2) -> list:
    """Backward-compatible entrypoint now using the stronger batting smoother."""
    return median_mean_smooth_coco_keypoints_sequence(kpts_sequence)
