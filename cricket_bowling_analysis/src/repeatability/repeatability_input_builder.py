"""Build repeatability input CSVs from frame-wise feature CSVs.

The source CSV is treated as read-only. This module copies its columns exactly,
adds minimal repeatability metadata if missing, and writes a new CSV under the
repeatability output tree.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import DEFAULT_OUTPUT_DIR, OUTPUT_SUBDIRS
from .feature_repair import repair_repeatability_features


def _default_output_path(frame_csv_path: Path) -> Path:
    return Path(DEFAULT_OUTPUT_DIR) / OUTPUT_SUBDIRS["input"] / f"{frame_csv_path.stem}_repeatability_input.csv"


def build_repeatability_input(
    frame_csv_path,
    output_csv_path: Optional[str] = None,
    bowler_id: Optional[str] = None,
    delivery_id: Optional[str] = None,
    video_id: Optional[str] = None,
) -> pd.DataFrame:
    """Create a repeatability input copy from a frame-wise CSV.

    Args:
        frame_csv_path: Existing frame-wise feature CSV from video analysis.
        output_csv_path: Optional destination. Defaults to outputs/repeatability/input.
        bowler_id: Optional metadata value if missing in the CSV.
        delivery_id: Optional metadata value if missing in the CSV.
        video_id: Optional metadata value if missing in the CSV.

    Returns:
        The new repeatability input DataFrame.
    """
    source_path = Path(frame_csv_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Frame CSV not found: {source_path}")

    df = pd.read_csv(source_path)

    if "frame_id" not in df.columns:
        df.insert(0, "frame_id", range(len(df)))
        print("[REPEATABILITY] Warning: frame_id missing; created sequential frame_id.")

    df = df.sort_values("frame_id").reset_index(drop=True)

    inferred_delivery = delivery_id or source_path.stem
    if "delivery_id" not in df.columns:
        df["delivery_id"] = inferred_delivery
    if "bowler_id" not in df.columns:
        df["bowler_id"] = bowler_id or "unknown_bowler"
    if "video_id" not in df.columns:
        df["video_id"] = video_id or source_path.stem

    if "phase" not in df.columns:
        df["phase"] = "unassigned"
    if "phase_id" not in df.columns:
        df["phase_id"] = -1
    if "normalized_phase_time" not in df.columns:
        df["normalized_phase_time"] = 0.0

    df, repaired = repair_repeatability_features(df)
    if repaired:
        print(f"[REPEATABILITY] Repaired/created {len(repaired)} feature columns for training.")

    out_path = Path(output_csv_path) if output_csv_path else _default_output_path(source_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[REPEATABILITY] Saved repeatability input: {out_path}")
    return df
