"""Suggest side-on bowling events and 7 repeatability phases.

These detections are deliberately treated as suggestions. Manual review can
override every event before the repeatability analysis uses the phases.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_OUTPUT_DIR, EVENT_NAMES, OUTPUT_SUBDIRS, PHASE_NAMES


def _smooth_numeric_columns(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        if col != "frame_id":
            out[col] = out[col].rolling(window=window, center=True, min_periods=1).mean()
    return out


def _col(df: pd.DataFrame, name: str) -> Optional[str]:
    return name if name in df.columns else None


def _clip(frame: int, min_frame: int, max_frame: int) -> int:
    return int(max(min_frame, min(frame, max_frame)))


def _subset(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return df[(df["frame_id"] >= start) & (df["frame_id"] <= end)]


def _idx_frame(subset: pd.DataFrame, column: str, mode: str) -> Optional[int]:
    if subset.empty or column not in subset.columns or subset[column].isna().all():
        return None
    idx = subset[column].idxmax() if mode == "max" else subset[column].idxmin()
    return int(subset.loc[idx, "frame_id"])


def _marker_frame(df: pd.DataFrame, column: str) -> Optional[int]:
    if column not in df.columns:
        return None
    marked = df[pd.to_numeric(df[column], errors="coerce").fillna(0) > 0]
    if marked.empty:
        return None
    return int(marked.iloc[0]["frame_id"])


def _ordered_events(events: Dict[str, int], min_frame: int, max_frame: int) -> Dict[str, int]:
    """Keep event order valid so phase windows are continuous and non-overlapping."""
    jump_start = _clip(events.get("jump_start", min_frame), min_frame, max_frame)
    bfc = _clip(events.get("bfc", jump_start + 12), min_frame, max_frame)
    ffc = _clip(events.get("ffc", bfc + 16), min_frame, max_frame)
    release = _clip(events.get("release", ffc + 18), min_frame, max_frame)
    follow_end = _clip(events.get("follow_through_end", release + 25), min_frame, max_frame)

    bfc = max(bfc, jump_start + 4)
    ffc = max(ffc, bfc + 6)
    release = max(release, ffc + 6)
    follow_end = max(follow_end, release + 6)

    if follow_end > max_frame:
        shift = follow_end - max_frame
        release = max(ffc + 3, release - shift)
        follow_end = max_frame
    if release > max_frame:
        release = max_frame
    if ffc >= release:
        ffc = max(jump_start + 5, release - 6)
    if bfc >= ffc:
        bfc = max(jump_start + 3, ffc - 6)
    if jump_start >= bfc:
        jump_start = max(min_frame, bfc - 4)

    events["jump_start"] = _clip(jump_start, min_frame, max_frame)
    events["bfc"] = _clip(bfc, min_frame, max_frame)
    events["ffc"] = _clip(ffc, min_frame, max_frame)
    events["release"] = _clip(release, min_frame, max_frame)
    events["follow_through_end"] = _clip(follow_end, min_frame, max_frame)
    events["jump_peak"] = _clip(events.get("jump_peak") or ((events["jump_start"] + events["bfc"]) // 2), min_frame, max_frame)
    return events


def _detect_events(df: pd.DataFrame, manual_events: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    min_frame = int(df["frame_id"].min())
    max_frame = int(df["frame_id"].max())
    events: Dict[str, int] = {}
    manual_events = {k.lower(): int(v) for k, v in (manual_events or {}).items()}

    marked_release = _marker_frame(df, "is_release_frame")
    if "release" in manual_events:
        events["release"] = _clip(manual_events["release"], min_frame, max_frame)
    elif marked_release is not None:
        events["release"] = _clip(marked_release, min_frame, max_frame)
    else:
        ffc_guess = manual_events.get("ffc", min_frame + int((max_frame - min_frame) * 0.65))
        search = _subset(df, ffc_guess, min(ffc_guess + 35, max_frame))
        if _col(df, "wrist_speed"):
            events["release"] = _idx_frame(search, "wrist_speed", "max")
        if not events.get("release") and _col(df, "bowling_wrist_y"):
            events["release"] = _idx_frame(search, "bowling_wrist_y", "min")
        events["release"] = _clip(events.get("release") or ffc_guess + 20, min_frame, max_frame)

    if "ffc" in manual_events:
        events["ffc"] = _clip(manual_events["ffc"], min_frame, max_frame)
    else:
        search_start = max(min_frame, events["release"] - 40)
        search_end = max(search_start, events["release"] - 3)
        search = _subset(df, search_start, search_end)
        events["ffc"] = _idx_frame(search, "front_ankle_y", "max") if _col(df, "front_ankle_y") else None
        events["ffc"] = _clip(events.get("ffc") or events["release"] - 20, min_frame, max_frame)

    if "bfc" in manual_events:
        events["bfc"] = _clip(manual_events["bfc"], min_frame, max_frame)
    else:
        search = _subset(df, max(min_frame, events["ffc"] - 40), max(min_frame, events["ffc"] - 3))
        events["bfc"] = _idx_frame(search, "back_ankle_y", "max") if _col(df, "back_ankle_y") else None
        events["bfc"] = _clip(events.get("bfc") or events["ffc"] - 20, min_frame, max_frame)

    if "jump_start" in manual_events:
        events["jump_start"] = _clip(manual_events["jump_start"], min_frame, max_frame)
    else:
        jump_start = events["bfc"] - 15
        if _col(df, "hip_center_y"):
            search = _subset(df, max(min_frame, events["bfc"] - 35), max(min_frame, events["bfc"] - 5)).sort_values("frame_id")
            vals = search["hip_center_y"].to_numpy()
            frames = search["frame_id"].to_numpy()
            if len(vals) >= 4:
                diff = np.diff(vals)
                for i in range(len(diff) - 2):
                    if diff[i] < 0 and diff[i + 1] < 0 and diff[i + 2] < 0:
                        jump_start = int(frames[i])
                        break
        events["jump_start"] = _clip(jump_start, min_frame, max_frame)

    if "jump_peak" in manual_events:
        events["jump_peak"] = _clip(manual_events["jump_peak"], min_frame, max_frame)
    else:
        search = _subset(df, events["jump_start"], events["bfc"])
        peak = _idx_frame(search, "hip_center_y", "min") if _col(df, "hip_center_y") else None
        events["jump_peak"] = _clip(peak or ((events["jump_start"] + events["bfc"]) // 2), min_frame, max_frame)

    if "follow_through_end" in manual_events:
        events["follow_through_end"] = _clip(manual_events["follow_through_end"], min_frame, max_frame)
    else:
        end = events["release"] + 25
        if _col(df, "hip_speed"):
            search = _subset(df, events["release"] + 10, min(events["release"] + 45, max_frame)).sort_values("frame_id")
            if not search.empty and not search["hip_speed"].isna().all():
                peak = search["hip_speed"].max()
                below = search[search["hip_speed"] < peak * 0.40]
                if not below.empty:
                    end = int(below.iloc[0]["frame_id"])
        events["follow_through_end"] = _clip(end, min_frame, max_frame)

    for name in EVENT_NAMES:
        if name in manual_events:
            events[name] = _clip(manual_events[name], min_frame, max_frame)

    return _ordered_events(events, min_frame, max_frame)


def build_phases_from_events(events: Dict[str, int], min_frame: int, max_frame: int, delivery_id: str) -> pd.DataFrame:
    """Build the 7 phase rows from confirmed/suggested event frames."""
    bfc, ffc, release = events["bfc"], events["ffc"], events["release"]
    jump_start = events["jump_start"]
    jump_peak = events.get("jump_peak")
    follow_end = events["follow_through_end"]
    bfc_pre = max(jump_start, bfc - 5)
    bfc_post = min(ffc - 1, bfc + 5)
    ffc_pre = max(bfc_post + 1, ffc - 5)
    ffc_post = min(release - 1, ffc + 5)
    specs = [
        ("approach", min_frame, jump_start - 1, jump_start, "window"),
        ("jump_bound", jump_start, bfc_pre - 1, jump_peak, "window"),
        ("bfc_window", bfc_pre, bfc_post, bfc, "event_window"),
        ("bfc_to_ffc", bfc_post + 1, ffc_pre - 1, None, "window"),
        ("ffc_window", ffc_pre, ffc_post, ffc, "event_window"),
        ("ffc_to_release", ffc_post + 1, release, release, "window"),
        ("follow_through", release + 1, max_frame, follow_end, "window"),
    ]
    rows = []
    cursor = min_frame
    for phase, start, end, event_frame, phase_type in specs:
        start = max(cursor, start)
        end = max(start, end)
        cursor = end + 1
        rows.append({
            "delivery_id": delivery_id,
            "phase": phase,
            "start_frame": _clip(start, min_frame, max_frame),
            "end_frame": _clip(max(start, end), min_frame, max_frame),
            "event_frame": event_frame,
            "phase_type": phase_type,
        })
    return pd.DataFrame(rows)


def detect_sideon_phases(
    repeatability_input_csv_path,
    output_phase_csv_path: Optional[str] = None,
    output_events_csv_path: Optional[str] = None,
    manual_events: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Suggest side-on events and phases from a repeatability input CSV."""
    input_path = Path(repeatability_input_csv_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Repeatability input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if "frame_id" not in df.columns:
        df.insert(0, "frame_id", range(len(df)))
        print("[REPEATABILITY] Warning: frame_id missing; using sequential ids.")
    df = df.sort_values("frame_id").reset_index(drop=True)
    smoothed = _smooth_numeric_columns(df, window=5)

    delivery_id = str(df["delivery_id"].iloc[0]) if "delivery_id" in df.columns else input_path.stem
    min_frame, max_frame = int(df["frame_id"].min()), int(df["frame_id"].max())
    events = _detect_events(smoothed, manual_events=manual_events)
    phases_df = build_phases_from_events(events, min_frame, max_frame, delivery_id)

    event_rows = []
    for name in EVENT_NAMES:
        event_rows.append({
            "delivery_id": delivery_id,
            "event_name": name,
            "frame_id": events.get(name),
            "confidence": 1.0 if manual_events and name in manual_events else 0.6,
            "detection_method": "manual_override" if manual_events and name in manual_events else "auto_suggestion",
        })
    events_df = pd.DataFrame(event_rows)

    phase_path = Path(output_phase_csv_path) if output_phase_csv_path else Path(DEFAULT_OUTPUT_DIR) / OUTPUT_SUBDIRS["auto_phases"] / f"{delivery_id}_auto_phases.csv"
    events_path = Path(output_events_csv_path) if output_events_csv_path else Path(DEFAULT_OUTPUT_DIR) / OUTPUT_SUBDIRS["auto_events"] / f"{delivery_id}_auto_events.csv"
    phase_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    phases_df.to_csv(phase_path, index=False)
    events_df.to_csv(events_path, index=False)
    print(f"[REPEATABILITY] Saved auto phases: {phase_path}")
    print(f"[REPEATABILITY] Saved auto events: {events_path}")
    print(f"[REPEATABILITY] Suggested phases: {', '.join(PHASE_NAMES)}")
    return phases_df, events_df
