"""Assign confirmed repeatability phases to each frame."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_OUTPUT_DIR, OUTPUT_SUBDIRS, PHASE_ID_MAP, PHASE_PRIORITY


def assign_phases_to_frames(
    sideon_feature_csv_path,
    confirmed_phase_csv_path,
    output_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Add phase, phase_id, and normalized_phase_time to every frame."""
    feature_path = Path(sideon_feature_csv_path)
    phase_path = Path(confirmed_phase_csv_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_path}")
    if not phase_path.exists():
        raise FileNotFoundError(f"Confirmed phase CSV not found: {phase_path}")

    df = pd.read_csv(feature_path)
    phases_df = pd.read_csv(phase_path)
    if "frame_id" not in df.columns:
        df.insert(0, "frame_id", range(len(df)))

    df["phase"] = "unassigned"
    df["phase_id"] = -1
    df["normalized_phase_time"] = 0.0

    for phase in PHASE_PRIORITY:
        for _, row in phases_df[phases_df["phase"] == phase].iterrows():
            start, end = int(row["start_frame"]), int(row["end_frame"])
            mask = (df["frame_id"] >= start) & (df["frame_id"] <= end)
            df.loc[mask, "phase"] = phase
            df.loc[mask, "phase_id"] = PHASE_ID_MAP.get(phase, -1)
            denom = max(end - start, 1)
            df.loc[mask, "normalized_phase_time"] = np.clip((df.loc[mask, "frame_id"] - start) / denom, 0, 1)

    unassigned = int((df["phase"] == "unassigned").sum())
    if unassigned:
        print(f"[REPEATABILITY] Warning: {unassigned} frames are unassigned.")

    delivery_id = str(df["delivery_id"].iloc[0]) if "delivery_id" in df.columns else feature_path.stem
    out_path = Path(output_csv_path) if output_csv_path else Path(DEFAULT_OUTPUT_DIR) / OUTPUT_SUBDIRS["phase_labelled"] / f"{delivery_id}_phase_labelled.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[REPEATABILITY] Saved phase-labelled CSV: {out_path}")
    return df
