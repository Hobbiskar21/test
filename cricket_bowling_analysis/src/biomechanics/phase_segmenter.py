"""
src/biomechanics/phase_segmenter.py
─────────────────────────────────────────────────────────────────────────────
Bowling phase segmentation — 4 phases, 3 biomechanical boundaries.

WHY THESE SIGNALS?
──────────────────
  Velocity = d(position)/dt  →  amplifies noise on blurry / compressed video.
  Wrist Y is a smooth arc regardless of video quality.  Its two extremes
  are unambiguous physical landmarks:

  B1  movement_start : sustained body displacement  → bowler leaves mark
  B2  gather_frame   : wrist Y **maximum**           → arm fully wound back
  B3  release_frame  : wrist Y **minimum**           → arm at apex = release

PHASES:
  RUN-UP         [0,  B1)
  LOAD-GATHER    [B1, B2)
  DELIVERY       [B2, B3)
  FOLLOW-THROUGH [B3, end)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_1d(data: ArrayLike, fill: float = 0.0) -> np.ndarray:
    return np.array([fill if v is None else float(v) for v in data], dtype=np.float64)


def _extract_y(positions: list, fill: float = 0.5) -> np.ndarray:
    """Pull Y coordinate from a list of (x,y[,z]) positions or Nones."""
    out = []
    for p in positions:
        if p is None:
            out.append(fill)
        else:
            a = np.asarray(p, dtype=np.float64).ravel()
            out.append(float(a[1]) if len(a) > 1 else fill)
    return np.array(out, dtype=np.float64)


def _smooth_signal(sig: np.ndarray, window: int) -> np.ndarray:
    """
    Savitzky-Golay (scipy) with moving-average fallback.
    SG preserves peak height — critical for finding Y extremes accurately.
    """
    w = max(3, int(window))
    if len(sig) < w:
        return sig.copy()
    try:
        from scipy.signal import savgol_filter
        w = w if w % 2 else w + 1
        return savgol_filter(sig, w, min(3, w - 1), mode="nearest")
    except ImportError:
        k = np.ones(w) / w
        return np.convolve(np.pad(sig, w // 2, "edge"), k, "valid")[: len(sig)]


# ─────────────────────────────────────────────────────────────────────────────
# B1 — movement start  (RUN-UP → LOAD-GATHER)
# ─────────────────────────────────────────────────────────────────────────────

def detect_jump_peak(ankle_positions: list, fps: float) -> int:
    """
    B1: first frame of sustained body movement.

    Despite the legacy name this detects *movement start*, not a jump peak.
    Uses ankle positions as a body-displacement proxy — ankles are still
    until the bowler physically pushes off their mark.

    Algorithm
    ─────────
    1. Compute mean ankle Y per frame → body position proxy.
    2. Smooth with 100 ms window (kills footstep noise).
    3. Compute per-frame displacement.
    4. Threshold = 70th-percentile of displacements (adapts to video scale).
    5. Return first frame of 3+ consecutive above-threshold frames.
    """
    n = len(ankle_positions) if ankle_positions else 0
    if n < 10:
        return max(1, int(n * 0.15))

    y_vals = []
    for frame in ankle_positions:
        if frame is None or len(frame) < 1:
            y_vals.append(0.5)
        else:
            ys = [float(a[1]) for a in frame if a is not None]
            y_vals.append(float(np.mean(ys)) if ys else 0.5)

    y = _smooth_signal(np.array(y_vals), max(3, int(fps * 0.10)))
    disp = np.abs(np.diff(y, prepend=y[0]))
    threshold = float(np.percentile(disp, 70))

    run = 0
    for i in range(1, n):
        if disp[i] > threshold:
            run += 1
            if run >= 3:
                return max(1, i - 2)
        else:
            run = 0
    return max(1, int(n * 0.15))


# ─────────────────────────────────────────────────────────────────────────────
# B2 — gather frame  (LOAD-GATHER → DELIVERY)
# ─────────────────────────────────────────────────────────────────────────────

def detect_arm_rotation_start(
    wrist_positions: list,
    hip_x_velocity: np.ndarray,
    fps: float,
    search_start: int = 0,
) -> int:
    """
    B2: wrist Y maximum = arm fully wound back = gather complete.

    Despite the legacy name this is NOT elbow-angle based.  Elbow angular
    velocity is noisy and fires too early.  Wrist Y max is the cleanest
    single-frame event in the entire bowling arc.

    Algorithm
    ─────────
    1. Search [B1, 72% of clip] — gather always precedes final third.
    2. Smooth wrist Y at 100 ms to kill run-up jitter.
    3. Global Y maximum in window = arm-back candidate.
    4. Confirm with nearest hip forward-velocity dip (foot plant ±0.3 s).
       If a dip exists, nudge boundary to midpoint for sub-frame accuracy.
    """
    n = len(wrist_positions)
    if n < 4:
        return max(1, search_start + 1)

    wy = _smooth_signal(_extract_y(wrist_positions), max(3, int(fps * 0.10)))
    end = int(np.clip(int(n * 0.72), search_start + 2, n - 2))
    candidate = search_start + int(np.argmax(wy[search_start:end]))

    # Confirm with hip velocity dip
    hw = max(1, int(fps * 0.30))
    lo, hi = max(search_start, candidate - hw), min(n - 1, candidate + hw)
    hip_win = hip_x_velocity[lo : hi + 1]
    if len(hip_win) > 0:
        dip = lo + int(np.argmin(hip_win))
        if abs(dip - candidate) <= hw:
            candidate = (candidate + dip) // 2

    return int(np.clip(candidate, search_start + 1, n - 2))


# ─────────────────────────────────────────────────────────────────────────────
# B3 — release frame  (DELIVERY → FOLLOW-THROUGH)
# ─────────────────────────────────────────────────────────────────────────────

def detect_ball_release(
    wrist_positions: list,
    fps: float,
    search_start: int = 0,
) -> int:
    """
    B3: wrist Y minimum = arm at highest point = ball leaves hand.

    This is the "right arm at highest position" landmark requested.
    Uses tight 50 ms smoothing to preserve the sharp arm-over peak
    without smearing it earlier.

    Note: wrist_velocities / elbow_angles params kept for API compat
    but the Y-minimum is strictly more reliable than a velocity peak
    on compressed sports footage.
    """
    n = len(wrist_positions)
    if n < 2:
        return max(search_start + 1, n - 1)

    wy = _smooth_signal(_extract_y(wrist_positions), max(3, int(fps * 0.05)))
    release = search_start + int(np.argmin(wy[search_start:]))
    return int(np.clip(release, search_start + 1, n - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Phase duration report
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(
    labels: np.ndarray,
    boundaries: dict,
    fps: float,
    path: Optional[str | Path],
) -> dict:
    """
    Build and optionally write phase_report.json.

    Schema
    ──────
    {
      "fps": 30, "total_frames": 450, "total_duration_s": 15.0,
      "boundaries": { ... },
      "phases": {
        "RUN-UP": {
          "start_frame": 0, "end_frame": 17,
          "frame_count": 18, "duration_s": 0.6, "pct": 4.0
        }, ...
      }
    }
    """
    n = len(labels)
    b1 = boundaries["movement_start_frame"]
    b2 = boundaries["jump_frame"]            # gather_frame
    b3 = boundaries["release_frame"]

    spans = [
        ("RUN-UP",         0,  b1),
        ("LOAD-GATHER",    b1, b2),
        ("DELIVERY",       b2, b3),
        ("FOLLOW-THROUGH", b3, n),
    ]
    phases = {
        name: {
            "start_frame": s,
            "end_frame":   e - 1,
            "frame_count": e - s,
            "duration_s":  round((e - s) / fps, 3),
            "pct":         round(100.0 * (e - s) / n, 1) if n else 0.0,
        }
        for name, s, e in spans
    }

    report = {
        "fps":              fps,
        "total_frames":     n,
        "total_duration_s": round(n / fps, 3),
        "boundaries":       boundaries,
        "phases":           phases,
    }

    if path is not None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[PHASE] Report → {p.resolve()}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def segment_phases(
    wrist_positions:   list,
    wrist_velocities:  list,
    hip_velocities:    list,
    knee_angles:       list,
    fps:               float,
    hip_positions:     list = None,
    ankle_positions:   list = None,
    elbow_angles:      list = None,
    phase_report_path: Optional[str | Path] = None,
) -> tuple[np.ndarray, dict]:
    """
    Segment a bowling delivery into four phases.

    Parameters
    ──────────
    wrist_positions   : list[(x,y)|None]  right wrist per frame
    wrist_velocities  : list[float|None]  scalar wrist speed   [unused; kept for compat]
    hip_velocities    : list[float|None]  horizontal hip speed (B2 confirmation)
    knee_angles       : list[float|None]  front knee angle     [unused; kept for compat]
    fps               : float
    hip_positions     : list[(x,y)|None]  [unused; kept for compat]
    ankle_positions   : list[[(lx,ly),(rx,ry)]|None]  used for B1 movement start
    elbow_angles      : list[float|None]  [unused; kept for compat]
    phase_report_path : optional path — writes phase_report.json if given

    Returns
    ───────
    labels  : np.ndarray[object] shape (N,)
    report  : dict  (boundaries + per-phase durations)
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    n = len(wrist_positions)
    if n == 0:
        raise ValueError("wrist_positions is empty.")

    hip_x_vel = _to_1d(hip_velocities)

    # ── B1 ───────────────────────────────────────────────────────────────────
    movement_start = detect_jump_peak(ankle_positions or [], fps)

    # ── B2 ───────────────────────────────────────────────────────────────────
    gather_frame = detect_arm_rotation_start(
        wrist_positions, hip_x_vel, fps, search_start=movement_start
    )

    # ── B3 ───────────────────────────────────────────────────────────────────
    release_frame = detect_ball_release(
        wrist_positions, fps, search_start=gather_frame
    )

    # ── Strict ordering guard ─────────────────────────────────────────────────
    movement_start = int(np.clip(movement_start, 1,                   n - 3))
    gather_frame   = int(np.clip(gather_frame,   movement_start + 1,  n - 2))
    release_frame  = int(np.clip(release_frame,  gather_frame + 1,    n - 1))

    if gather_frame <= movement_start:
        warnings.warn(f"gather_frame={gather_frame} ≤ movement_start={movement_start}; clamped.", UserWarning)
        gather_frame = movement_start + 1

    if release_frame <= gather_frame:
        warnings.warn(f"release_frame={release_frame} ≤ gather_frame={gather_frame}; clamped.", UserWarning)
        release_frame = gather_frame + 1

    # ── Labels ────────────────────────────────────────────────────────────────
    labels = np.empty(n, dtype=object)
    labels[:movement_start]                     = "RUN-UP"
    labels[movement_start : gather_frame]       = "LOAD-GATHER"
    labels[gather_frame   : release_frame]      = "DELIVERY"
    labels[release_frame  :]                    = "FOLLOW-THROUGH"

    boundaries = {
        "movement_start_frame":    movement_start,
        "jump_frame":              gather_frame,        # B2 — legacy alias
        "arm_rotation_start_frame": gather_frame,       # B2 — legacy alias
        "release_frame":           release_frame,       # B3
        "peak_wrist_frame":        release_frame,       # B3 — legacy alias
    }

    # ── Report ────────────────────────────────────────────────────────────────
    report = _write_report(labels, boundaries, fps, phase_report_path)

    print(
        f"[PHASE] B1={movement_start} B2={gather_frame} B3={release_frame} | "
        + "  ".join(f"{p}:{int(np.sum(labels==p))}f"
                    for p in ["RUN-UP","LOAD-GATHER","DELIVERY","FOLLOW-THROUGH"])
    )

    return labels, report


def phase_frame_ranges(phase_map) -> dict:
    """Convert frame-level phase labels to {phase: {start, end}} ranges."""
    ranges = {}
    items = phase_map.items() if isinstance(phase_map, dict) else enumerate(phase_map)
    for idx, phase in sorted(items, key=lambda x: x[0]):
        if phase not in ranges:
            ranges[phase] = {"start": idx, "end": idx}
        else:
            ranges[phase]["end"] = idx
    return ranges
