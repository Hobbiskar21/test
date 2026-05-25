"""Repair repeatability feature tables so training never sees null features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LSTM_FEATURES, METADATA_COLUMNS, PHASE_FEATURES, SIDEON_FEATURES


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _derive_velocity(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    x = _numeric(df, x_col).ffill().bfill().fillna(0.0)
    y = _numeric(df, y_col).ffill().bfill().fillna(0.0)
    return np.sqrt(x.diff().fillna(0.0) ** 2 + y.diff().fillna(0.0) ** 2)


def _diff(df: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric(df, column).ffill().bfill().fillna(0.0)
    return values.diff().fillna(0.0)


def repair_repeatability_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create every repeatability feature column and replace nulls with numbers.

    Returns the repaired DataFrame and a list of columns that had to be created.
    """
    out = df.copy()
    if "frame_id" not in out.columns:
        out.insert(0, "frame_id", range(len(out)))
    if "frame" not in out.columns:
        out["frame"] = out["frame_id"]
    if "delivery_id" not in out.columns:
        out["delivery_id"] = "unknown_delivery"
    if "bowler_id" not in out.columns:
        out["bowler_id"] = "unknown_bowler"
    if "video_id" not in out.columns:
        out["video_id"] = out["delivery_id"]

    created: list[str] = []

    aliases = {
        "trunk_lean_angle": ["trunk_lean", "trunk_flexion"],
        "bowling_elbow_angle": ["elbow_angle"],
        "bowling_arm_angle": ["shoulder_angle"],
        "front_leg_angle": ["front_knee_angle"],
        "back_leg_angle": ["back_knee_angle"],
    }
    for target, sources in aliases.items():
        if target not in out.columns:
            for source in sources:
                if source in out.columns:
                    out[target] = out[source]
                    created.append(target)
                    break

    derived = {
        "front_knee_angle_velocity": lambda d: _diff(d, "front_knee_angle"),
        "arm_angle_velocity": lambda d: _diff(d, "bowling_arm_angle"),
        "trunk_lean_velocity": lambda d: _diff(d, "trunk_lean_angle"),
        "head_speed": lambda d: _derive_velocity(d, "head_x", "head_y"),
        "front_ankle_speed": lambda d: _derive_velocity(d, "front_ankle_x", "front_ankle_y"),
        "stride_length_proxy": lambda d: np.sqrt(
            (_numeric(d, "front_ankle_x").ffill().bfill().fillna(0.0) - _numeric(d, "back_ankle_x").ffill().bfill().fillna(0.0)) ** 2
            + (_numeric(d, "front_ankle_y").ffill().bfill().fillna(0.0) - _numeric(d, "back_ankle_y").ffill().bfill().fillna(0.0)) ** 2
        ),
        "release_height_proxy": lambda d: _numeric(d, "bowling_wrist_y"),
    }
    for column, builder in derived.items():
        if column not in out.columns:
            out[column] = builder(out)
            created.append(column)

    requested = sorted(set(SIDEON_FEATURES).union(LSTM_FEATURES).union(*(PHASE_FEATURES[p] for p in PHASE_FEATURES)))
    for column in requested:
        if column not in out.columns:
            out[column] = 0.0
            created.append(column)

    text_columns = {"session_id", "delivery_id", "bowler_id", "video_id", "phase"}
    for column in out.columns:
        if column in text_columns:
            out[column] = out[column].fillna("")
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce").ffill().bfill().fillna(0.0)

    for column in METADATA_COLUMNS:
        if column in out.columns:
            out[column] = out[column].fillna("")

    if "phase" not in out.columns:
        out["phase"] = "unassigned"
    if "phase_id" not in out.columns:
        out["phase_id"] = -1
    if "normalized_phase_time" not in out.columns:
        out["normalized_phase_time"] = 0.0

    out["phase"] = out["phase"].fillna("unassigned")
    out["phase_id"] = pd.to_numeric(out["phase_id"], errors="coerce").fillna(-1).astype(int)
    out["normalized_phase_time"] = pd.to_numeric(out["normalized_phase_time"], errors="coerce").fillna(0.0)

    return out, sorted(set(created))
