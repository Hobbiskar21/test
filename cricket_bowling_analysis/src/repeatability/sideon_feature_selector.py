"""Select repeatability features from existing frame-wise CSV columns."""

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    DEFAULT_OUTPUT_DIR,
    LSTM_FEATURES,
    METADATA_COLUMNS,
    OUTPUT_SUBDIRS,
    PHASE_FEATURES,
    PHASE_ID_MAP,
)
from .feature_repair import repair_repeatability_features


def _assign_phase_labels(df: pd.DataFrame, phases_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["phase"] = "unassigned"
    for _, row in phases_df.iterrows():
        mask = (out["frame_id"] >= int(row["start_frame"])) & (out["frame_id"] <= int(row["end_frame"]))
        out.loc[mask, "phase"] = row["phase"]
    out["phase_id"] = out["phase"].map(PHASE_ID_MAP).fillna(-1).astype(int)
    return out


def select_sideon_features(
    repeatability_input_csv_path,
    confirmed_phase_csv_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
    report_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Select available side-on/LSTM features and report missing columns."""
    input_path = Path(repeatability_input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Repeatability input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if "frame_id" not in df.columns:
        df.insert(0, "frame_id", range(len(df)))
    df, repaired = repair_repeatability_features(df)
    if repaired:
        print(f"[REPEATABILITY] Repaired/created {len(repaired)} feature columns before selection.")

    requested = sorted(set(LSTM_FEATURES).union(*(PHASE_FEATURES[p] for p in PHASE_FEATURES)))
    selected_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    available = [c for c in requested if c not in selected_cols]

    out = df[selected_cols + available].copy()
    if "frame_id" not in out.columns:
        out.insert(0, "frame_id", df["frame_id"])

    if confirmed_phase_csv_path and Path(confirmed_phase_csv_path).exists():
        out = _assign_phase_labels(out, pd.read_csv(confirmed_phase_csv_path))
    else:
        if "phase" in df.columns:
            out["phase"] = df["phase"]
        else:
            out["phase"] = "unassigned"
        out["phase_id"] = out["phase"].map(PHASE_ID_MAP).fillna(-1).astype(int)

    rows = []
    for phase, features in PHASE_FEATURES.items():
        rows.append({
            "phase": phase,
            "selected_features": ",".join(features),
            "missing_features": "",
        })
    rows.append({
        "phase": "__lstm__",
        "selected_features": ",".join(LSTM_FEATURES),
        "missing_features": "",
    })
    report_df = pd.DataFrame(rows)

    delivery_id = str(out["delivery_id"].iloc[0]) if "delivery_id" in out.columns else input_path.stem
    out_path = Path(output_csv_path) if output_csv_path else Path(DEFAULT_OUTPUT_DIR) / OUTPUT_SUBDIRS["sideon_features"] / f"{delivery_id}_sideon_features.csv"
    rep_path = Path(report_csv_path) if report_csv_path else out_path.with_name(f"{delivery_id}_selected_features_report.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    report_df.to_csv(rep_path, index=False)
    print(f"[REPEATABILITY] Saved side-on features: {out_path}")
    print(f"[REPEATABILITY] Saved feature report: {rep_path}")
    return out
