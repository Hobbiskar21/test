"""
src/biomechanics/runup_analyser.py
-----------------------------------
Analyzes bowler's run-up phase and generates comprehensive visual report.

4 panels:
    1. Momentum build-up — from first frame to last frame (full delivery)
    2. Hip velocity — during run-up phase only
    3. Gate pattern — ankle separation during run-up
    4. Metrics summary table
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# COCO landmark indices (17 keypoints)
L_HIP      = 11
R_HIP      = 12
L_ANKLE    = 15
R_ANKLE    = 16
L_SHOULDER = 5
R_SHOULDER = 6


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_hip_path(landmarks_sequence, phase_map, width, height, phase_filter=None):
    """
    Extract hip midpoint (x, y) for every frame.

    Args:
        phase_filter : if given (e.g. "RUN-UP"), only extract that phase.
                       if None, extract ALL frames.

    Returns:
        hip_path    : list of (x, y) or None per extracted frame
        frame_indices: list of frame indices that were extracted
    """
    hip_path      = []
    frame_indices = []

    for frame_idx, lm in enumerate(landmarks_sequence):
        # Phase filter
        if phase_filter is not None:
            if phase_map.get(frame_idx) != phase_filter:
                continue

        frame_indices.append(frame_idx)

        if lm is None:
            hip_path.append(None)
            continue

        l_hip = lm[L_HIP]
        r_hip = lm[R_HIP]

        if l_hip.visibility >= 0.5 and r_hip.visibility >= 0.5:
            hip_x = (l_hip.x + r_hip.x) / 2 * width
            hip_y = (l_hip.y + r_hip.y) / 2 * height
            hip_path.append((hip_x, hip_y))
        else:
            hip_path.append(None)

    return hip_path, frame_indices


def _calculate_velocities(positions, fps):
    """
    Calculate speed per frame using central differences.
    v[t] = |pos[t+1] - pos[t-1]| / 2

    Handles None positions — returns 0.0 for those frames.

    Returns:
        list of float speeds, same length as positions
    """
    n          = len(positions)
    velocities = []

    for i in range(n):
        prev = positions[i - 1] if i > 0     else None
        nxt  = positions[i + 1] if i < n - 1 else None
        curr = positions[i]

        if curr is None or prev is None or nxt is None:
            velocities.append(0.0)
            continue

        dx    = nxt[0] - prev[0]
        dy    = nxt[1] - prev[1]
        speed = np.sqrt(dx**2 + dy**2) / 2.0
        velocities.append(float(speed))

    return velocities


def _calculate_momentum(velocities):
    """
    Cumulative sum of velocities = momentum proxy.
    Starts at 0, builds up as bowler accelerates.
    Tracks from whatever the first frame is to the last.
    """
    return np.cumsum(velocities).tolist()


def _find_foot_contacts(landmarks_sequence, runup_frames, velocity_drop_threshold=0.3):
    """
    Find backfoot and frontfoot contact frames using ankle Y position drops.

    Backfoot contact = right ankle Y drops sharply (foot plants on ground)
    Frontfoot contact = left ankle Y drops sharply after backfoot

    Returns:
        backfoot_frame, frontfoot_frame — frame indices or None
    """
    ankle_y_l = []
    ankle_y_r = []

    for frame_idx in runup_frames:
        lm = landmarks_sequence[frame_idx]
        if lm is None:
            ankle_y_l.append(np.nan)
            ankle_y_r.append(np.nan)
            continue

        l_a = lm[L_ANKLE]
        r_a = lm[R_ANKLE]

        ankle_y_l.append(l_a.y if l_a.visibility >= 0.5 else np.nan)
        ankle_y_r.append(r_a.y if r_a.visibility >= 0.5 else np.nan)

    backfoot_frame  = None
    frontfoot_frame = None

    # Right ankle — backfoot contact
    r_arr    = np.array(ankle_y_r)
    inverted = -r_arr
    peaks, _ = find_peaks(np.nan_to_num(inverted), height=velocity_drop_threshold)
    if len(peaks) > 0:
        backfoot_frame = runup_frames[peaks[0]]

    # Left ankle — frontfoot contact (search after backfoot)
    if backfoot_frame is not None:
        try:
            bf_local = runup_frames.index(backfoot_frame) + 1
        except ValueError:
            bf_local = 0
        l_arr    = np.array(ankle_y_l[bf_local:])
        inverted = -l_arr
        peaks, _ = find_peaks(np.nan_to_num(inverted), height=velocity_drop_threshold)
        if len(peaks) > 0:
            frontfoot_frame = runup_frames[bf_local + peaks[0]]

    return backfoot_frame, frontfoot_frame


def _calculate_gate_pattern_all(landmarks_sequence, width):
    """
    Calculate ankle separation for ALL frames (not just run-up).
    Gate width = horizontal distance between left and right ankles.
    """
    gate_widths = []

    for lm in landmarks_sequence:
        if lm is None:
            gate_widths.append(None)
            continue

        l_a = lm[L_ANKLE]
        r_a = lm[R_ANKLE]

        if l_a.visibility >= 0.5 and r_a.visibility >= 0.5:
            gate_widths.append(abs(r_a.x - l_a.x) * width)
        else:
            gate_widths.append(None)

    return gate_widths


def _calculate_shoulder_width(landmarks_sequence, width):
    """Average shoulder width from first 10 valid frames — used as gate reference."""
    widths = []
    for lm in landmarks_sequence[:10]:
        if lm is None:
            continue
        l_s = lm[L_SHOULDER]
        r_s = lm[R_SHOULDER]
        if l_s.visibility >= 0.5 and r_s.visibility >= 0.5:
            widths.append(abs(r_s.x - l_s.x) * width)
    return float(np.mean(widths)) if widths else 0.0


def _calculate_approach_angle(hip_path_runup, backfoot_frame, runup_frames):
    """
    Fit a line through the last 15 hip positions before backfoot contact.
    Angle of that line relative to vertical = approach angle.
    """
    if backfoot_frame is None or len(hip_path_runup) < 2:
        return 0.0

    try:
        bf_idx = runup_frames.index(backfoot_frame)
    except ValueError:
        return 0.0

    start    = max(0, bf_idx - 15)
    segment  = [p for p in hip_path_runup[start:bf_idx + 1] if p is not None]

    if len(segment) < 2:
        return 0.0

    xs      = np.array([p[0] for p in segment])
    ys      = np.array([p[1] for p in segment])
    coeffs  = np.polyfit(xs, ys, 1)
    angle   = abs(np.degrees(np.arctan(coeffs[0])))
    return float(min(angle, 90.0))


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyse_runup(
    landmarks_sequence,
    phase_map,
    fps,
    width,
    height,
    output_path,
    velocity_drop_threshold=0.3,
    gate_warn_multiplier=1.5,
    camera_angle_type=None,
):
    """
    Analyse bowler's run-up and generate 4-panel visual report as PNG.

    Parameters
    ----------
    landmarks_sequence : list — per-frame COCO landmarks (17 keypoints)
    phase_map          : dict — frame_idx → phase string
    fps                : float
    width, height      : int — frame dimensions
    output_path        : str — where to save PNG
    velocity_drop_threshold : float
    gate_warn_multiplier    : float
    camera_angle_type       : str or None — skip viz if 'side_on'

    Returns
    -------
    dict of metrics
    """
    # ── Extract hip path for RUN-UP phase (for gate, angle, foot contacts) ──
    hip_path_runup, runup_frames = _extract_hip_path(
        landmarks_sequence, phase_map, width, height, phase_filter="RUN-UP"
    )

    # ── Extract hip path for ALL frames (for full momentum curve) ───────────
    hip_path_all, all_frames = _extract_hip_path(
        landmarks_sequence, phase_map, width, height, phase_filter=None
    )

    if len(runup_frames) < 2:
        print("[RUNUP] Not enough RUN-UP frames to analyse.")
        return _empty_metrics()

    # ── Velocities ────────────────────────────────────────────────────────────
    # Run-up only — for hip velocity panel
    velocities_runup = _calculate_velocities(hip_path_runup, fps)

    # All frames — for full momentum curve
    velocities_all   = _calculate_velocities(hip_path_all, fps)

    # ── Momentum (full delivery, start to end) ────────────────────────────────
    # This tracks momentum from frame 0 all the way to the last frame
    # so we can see if peak momentum is before or after release
    momentum_all = _calculate_momentum(velocities_all)

    # ── Foot contacts ─────────────────────────────────────────────────────────
    backfoot_frame, frontfoot_frame = _find_foot_contacts(
        landmarks_sequence, runup_frames, velocity_drop_threshold
    )

    # ── Gate pattern ──────────────────────────────────────────────────────────
    gate_widths_all = _calculate_gate_pattern_all(landmarks_sequence, width)
    shoulder_width = _calculate_shoulder_width(landmarks_sequence, width)

    # ── Approach angle ────────────────────────────────────────────────────────
    approach_angle = _calculate_approach_angle(hip_path_runup, backfoot_frame, runup_frames)

    # ── Find release frame from phase_map ─────────────────────────────────────
    release_frame = None
    for frame_idx in sorted(phase_map.keys()):
        if phase_map.get(frame_idx) in ("DELIVERY", "FOLLOW-THROUGH"):
            release_frame = frame_idx
            break

    # ── Peak momentum across ALL frames ───────────────────────────────────────
    peak_momentum_idx   = int(np.argmax(momentum_all))
    peak_momentum_frame = all_frames[peak_momentum_idx] if peak_momentum_idx < len(all_frames) else None

    # ── Momentum at backfoot contact ──────────────────────────────────────────
    momentum_at_backfoot = 0.0
    if backfoot_frame is not None and backfoot_frame in all_frames:
        bf_all_idx           = all_frames.index(backfoot_frame)
        momentum_at_backfoot = float(momentum_all[bf_all_idx])

    # ── Momentum at release ───────────────────────────────────────────────────
    momentum_at_release = 0.0
    if release_frame is not None and release_frame in all_frames:
        rel_all_idx         = all_frames.index(release_frame)
        momentum_at_release = float(momentum_all[rel_all_idx])

    # ── Peak velocity ─────────────────────────────────────────────────────────
    peak_velocity     = float(max(velocities_runup)) if velocities_runup else 0.0
    peak_vel_local    = int(np.argmax(velocities_runup)) if velocities_runup else 0
    peak_vel_frame    = runup_frames[peak_vel_local] if peak_vel_local < len(runup_frames) else None

    # ── Gate stats ────────────────────────────────────────────────────────────
    gate_clean   = [w for w in gate_widths_all if w is not None]
    average_gate = float(np.mean(gate_clean)) if gate_clean else 0.0
    stride_count = len(find_peaks(np.nan_to_num([w if w else 0 for w in gate_widths_all]),
                                   height=50)[0])

    # ── Visualisation ─────────────────────────────────────────────────────────
    try:
        if camera_angle_type != "side_on":
            _create_visualization(
                all_frames       = all_frames,
                momentum_all     = momentum_all,
                runup_frames     = runup_frames,
                velocities_runup = velocities_runup,
                velocities_all   = velocities_all,
                gate_widths_all  = gate_widths_all,
                shoulder_width   = shoulder_width,
                backfoot_frame   = backfoot_frame,
                frontfoot_frame  = frontfoot_frame,
                peak_momentum_frame = peak_momentum_frame,
                release_frame    = release_frame,
                approach_angle   = approach_angle,
                average_gate     = average_gate,
                stride_count     = stride_count,
                peak_velocity    = peak_velocity,
                output_path      = output_path,
                gate_warn_multiplier = gate_warn_multiplier,
                phase_map        = phase_map,
            )
        else:
            print("[RUNUP] Side-on camera — skipping visualisation.")
    except Exception as e:
        print(f"[RUNUP] Error creating visualization: {e}")
        print(f"[RUNUP] Attempting fallback PNG creation to: {output_path}")
        try:
            _create_visualization(
                all_frames       = all_frames,
                momentum_all     = momentum_all,
                runup_frames     = runup_frames,
                velocities_runup = velocities_runup,
                velocities_all   = velocities_all,
                gate_widths_all  = gate_widths_all,
                shoulder_width   = shoulder_width,
                backfoot_frame   = backfoot_frame,
                frontfoot_frame  = frontfoot_frame,
                peak_momentum_frame = peak_momentum_frame,
                release_frame    = release_frame,
                approach_angle   = approach_angle,
                average_gate     = average_gate,
                stride_count     = stride_count,
                peak_velocity    = peak_velocity,
                output_path      = output_path,
                gate_warn_multiplier = gate_warn_multiplier,
                phase_map        = phase_map,
            )
        except Exception as e2:
            print(f"[RUNUP] Fallback also failed: {e2}")

    return {
        "peak_momentum_frame":     peak_momentum_frame,
        "backfoot_contact_frame":  backfoot_frame,
        "frontfoot_contact_frame": frontfoot_frame,
        "approach_angle_deg":      approach_angle,
        "average_gate_width_px":   average_gate,
        "stride_count":            stride_count,
        "peak_velocity_px_frame":  peak_velocity,
        "momentum_at_backfoot":    momentum_at_backfoot,
        "momentum_at_release":     momentum_at_release,
    }


def _empty_metrics():
    return {
        "peak_momentum_frame":     None,
        "backfoot_contact_frame":  None,
        "frontfoot_contact_frame": None,
        "approach_angle_deg":      0.0,
        "average_gate_width_px":   0.0,
        "stride_count":            0,
        "peak_velocity_px_frame":  0.0,
        "momentum_at_backfoot":    0.0,
        "momentum_at_release":     0.0,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _create_visualization(
    all_frames, momentum_all, runup_frames, velocities_runup, velocities_all,
    gate_widths_all, shoulder_width, backfoot_frame, frontfoot_frame,
    peak_momentum_frame, release_frame, approach_angle, average_gate,
    stride_count, peak_velocity, output_path, gate_warn_multiplier, phase_map,
):
    """
    4-panel portrait figure:
        Panel 1 — Full momentum curve (frame 0 → last frame)
        Panel 2 — Hip velocity during RUN-UP phase only
        Panel 3 — Gate pattern during RUN-UP
        Panel 4 — Metrics summary table
    """
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 22))

    ax1 = plt.subplot(4, 1, 1)
    ax2 = plt.subplot(4, 1, 2)
    ax3 = plt.subplot(4, 1, 3)
    ax4 = plt.subplot(4, 1, 4)

    _panel_momentum(ax1, all_frames, momentum_all, phase_map,
                    backfoot_frame, frontfoot_frame, release_frame,
                    peak_momentum_frame)

    _panel_hip_velocity(ax2, all_frames, velocities_all,
                        backfoot_frame, frontfoot_frame, peak_velocity, phase_map)

    _panel_gate(ax3, all_frames, gate_widths_all,
                shoulder_width, gate_warn_multiplier, phase_map)

    _panel_metrics(ax4, backfoot_frame, frontfoot_frame, approach_angle,
                   shoulder_width, average_gate, stride_count,
                   peak_velocity, peak_momentum_frame, release_frame,
                   momentum_all, all_frames)

    plt.tight_layout(pad=3.0)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"[RUNUP] Visual saved to {output_path}")


# ── Panel 1: Full momentum ────────────────────────────────────────────────────

def _panel_momentum(ax, all_frames, momentum_all, phase_map,
                    backfoot_frame, frontfoot_frame, release_frame,
                    peak_momentum_frame):
    """
    Momentum from first frame to last frame of the full delivery.
    Coloured by phase so you can see momentum at each stage.
    Peak momentum labelled red (pre-release) or orange (post-release).
    """
    ax.set_title("Momentum Build-up — Full Delivery (Start → End)",
                 fontsize=13, color="white", fontweight="bold")

    # Phase colour map
    phase_colors = {
        "RUN-UP":          "#4488ff",   # blue
        "LOAD-UP":         "#ffdd00",   # yellow
        "LOAD-GATHER":     "#ffdd00",   # yellow (alias)
        "DELIVERY":        "#ff4444",   # red
        "FOLLOW-THROUGH":  "#44dd88",   # green
    }

    # Draw coloured segments
    for i in range(len(all_frames) - 1):
        phase = phase_map.get(all_frames[i], "RUN-UP")
        color = phase_colors.get(phase, "#888888")
        ax.plot([all_frames[i], all_frames[i + 1]],
                [momentum_all[i], momentum_all[i + 1]],
                color=color, linewidth=3, alpha=0.9, zorder=4)

    # Semi-transparent fill under curve
    ax.fill_between(all_frames, momentum_all, alpha=0.15, color="cyan", zorder=2)

    max_mom = max(momentum_all) if momentum_all else 1

    # Vertical markers helper
    def _vline(frame, color, label, y_frac=0.95):
        if frame is None:
            return
        ax.axvline(frame, color=color, linestyle="--", linewidth=1.8,
                   alpha=0.8, zorder=5)
        ax.text(frame + 0.3, max_mom * y_frac, label, color=color,
                fontsize=8, rotation=90, va="top", fontweight="bold")

    _vline(backfoot_frame,  "#4488ff", "Backfoot",   0.95)
    _vline(frontfoot_frame, "#44dd88", "Frontfoot",  0.88)
    _vline(release_frame,   "#ff44ff", "Release",    0.80)

    # Peak momentum — special label
    if peak_momentum_frame is not None and peak_momentum_frame in all_frames:
        pm_idx    = all_frames.index(peak_momentum_frame)
        pm_val    = momentum_all[pm_idx]

        if release_frame is not None and peak_momentum_frame > release_frame:
            pm_label = "PEAK MOMENTUM\n(post release)"
            pm_color = "orange"
        elif release_frame is not None and peak_momentum_frame < release_frame:
            pm_label = "PEAK MOMENTUM\n(pre release WARNING)"
            pm_color = "red"
        else:
            pm_label = "PEAK MOMENTUM"
            pm_color = "white"

        ax.axvline(peak_momentum_frame, color="white", linestyle="--",
                   linewidth=2.2, alpha=0.95, zorder=6)
        ax.text(peak_momentum_frame, pm_val * 1.02, pm_label,
                color=pm_color, fontsize=9, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="black",
                          alpha=0.85, edgecolor=pm_color, linewidth=1.5),
                zorder=7)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4488ff", label="RUN-UP"),
        Patch(facecolor="#ffdd00", label="LOAD-UP / GATHER"),
        Patch(facecolor="#ff4444", label="DELIVERY"),
        Patch(facecolor="#44dd88", label="FOLLOW-THROUGH"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              fontsize=8, framealpha=0.6)

    ax.set_xlabel("Frame Number", color="white", fontsize=10)
    ax.set_ylabel("Cumulative Momentum (px/frame)", color="white", fontsize=10)
    ax.grid(True, alpha=0.25, color="white")


# ── Panel 2: Hip velocity ─────────────────────────────────────────────────────

def _panel_hip_velocity(ax, all_frames, velocities_all,
                        backfoot_frame, frontfoot_frame, peak_velocity, phase_map=None):
    """
    Hip speed frame by frame across FULL DELIVERY (all frames).
    Shows how the bowler accelerates from start to finish.
    Peak velocity marked clearly.
    Coloured by phase if phase_map provided.
    """
    ax.set_title("Hip Velocity — Full Delivery",
                 fontsize=13, color="white", fontweight="bold")

    # Phase colour map
    phase_colors = {
        "RUN-UP":          "#4488ff",   # blue
        "LOAD-UP":         "#ffdd00",   # yellow
        "LOAD-GATHER":     "#ffdd00",   # yellow (alias)
        "DELIVERY":        "#ff4444",   # red
        "FOLLOW-THROUGH":  "#44dd88",   # green
    }

    # Draw coloured segments if phase_map available
    if phase_map is not None:
        for i in range(len(all_frames) - 1):
            phase = phase_map.get(all_frames[i], "RUN-UP")
            color = phase_colors.get(phase, "#888888")
            ax.plot([all_frames[i], all_frames[i + 1]],
                    [velocities_all[i], velocities_all[i + 1]],
                    color=color, linewidth=3, alpha=0.9, zorder=4)
        ax.fill_between(all_frames, velocities_all,
                        alpha=0.15, color="#00ccff", zorder=2)
    else:
        # Fallback to single color if no phase_map
        ax.plot(all_frames, velocities_all,
                color="#00ccff", linewidth=2, label="Hip Velocity", zorder=3)
        ax.fill_between(all_frames, velocities_all,
                        alpha=0.2, color="#00ccff", zorder=2)

    max_vel = max(velocities_all) if velocities_all else 1

    # Mark peak velocity
    if velocities_all:
        peak_idx   = int(np.argmax(velocities_all))
        peak_frame = all_frames[peak_idx]
        ax.axvline(peak_frame, color="yellow", linestyle="--",
                   linewidth=1.8, alpha=0.8, zorder=5)
        ax.text(peak_frame + 0.3, max_vel * 0.92, f"Peak\n{peak_velocity:.1f}px/f",
                color="yellow", fontsize=8, rotation=90, va="top", fontweight="bold")

    # Mark backfoot contact
    if backfoot_frame is not None and backfoot_frame in all_frames:
        ax.axvline(backfoot_frame, color="#4488ff", linestyle="--",
                   linewidth=1.8, alpha=0.8, zorder=5)
        ax.text(backfoot_frame + 0.3, max_vel * 0.80, "Backfoot",
                color="#4488ff", fontsize=8, rotation=90, va="top", fontweight="bold")

    # Mark frontfoot contact
    if frontfoot_frame is not None and frontfoot_frame in all_frames:
        ax.axvline(frontfoot_frame, color="#44dd88", linestyle="--",
                   linewidth=1.8, alpha=0.8, zorder=5)
        ax.text(frontfoot_frame + 0.3, max_vel * 0.70, "Frontfoot",
                color="#44dd88", fontsize=8, rotation=90, va="top", fontweight="bold")

    ax.set_xlabel("Frame Number", color="white", fontsize=10)
    ax.set_ylabel("Speed (pixels / frame)", color="white", fontsize=10)
    ax.grid(True, alpha=0.25, color="white")


# ── Panel 3: Gate pattern ─────────────────────────────────────────────────────

def _panel_gate(ax, all_frames, gate_widths_all, shoulder_width, gate_warn_multiplier, phase_map=None):
    """Ankle separation per frame — how wide the bowler's gate is across full delivery."""
    ax.set_title("Gate Pattern (Ankle Separation — Full Delivery)",
                 fontsize=13, color="white", fontweight="bold")

    # Phase colour map
    phase_colors = {
        "RUN-UP":          "#4488ff",   # blue
        "LOAD-UP":         "#ffdd00",   # yellow
        "LOAD-GATHER":     "#ffdd00",   # yellow (alias)
        "DELIVERY":        "#ff4444",   # red
        "FOLLOW-THROUGH":  "#44dd88",   # green
    }

    clean = [w if w is not None else np.nan for w in gate_widths_all]

    # Draw coloured segments if phase_map available
    if phase_map is not None:
        for i in range(len(all_frames) - 1):
            if not np.isnan(clean[i]) and not np.isnan(clean[i + 1]):
                phase = phase_map.get(all_frames[i], "RUN-UP")
                color = phase_colors.get(phase, "#888888")
                ax.plot([all_frames[i], all_frames[i + 1]],
                        [clean[i], clean[i + 1]],
                        color=color, linewidth=3, alpha=0.9, zorder=4)
    else:
        # Fallback to single color
        ax.plot(all_frames, clean, color="#dd44ff",
                linewidth=2, label="Gate Width", zorder=3)

    if shoulder_width > 0:
        ax.axhline(shoulder_width, color="yellow", linestyle="--",
                   linewidth=1.8, alpha=0.8,
                   label=f"Shoulder Width ({shoulder_width:.0f}px)")
        warn = shoulder_width * gate_warn_multiplier
        ax.axhline(warn, color="red", linestyle=":",
                   linewidth=1.5, alpha=0.6,
                   label=f"Warning ({warn:.0f}px)")
        ax.fill_between(all_frames, warn,
                        max(c for c in clean if not np.isnan(c)) * 1.1
                        if any(not np.isnan(c) for c in clean) else warn * 1.1,
                        alpha=0.08, color="red")

    ax.set_xlabel("Frame Number", color="white", fontsize=10)
    ax.set_ylabel("Ankle Separation (pixels)", color="white", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
    ax.grid(True, alpha=0.25, color="white")


# ── Panel 4: Metrics table ────────────────────────────────────────────────────

def _panel_metrics(ax, backfoot_frame, frontfoot_frame, approach_angle,
                   shoulder_width, average_gate, stride_count,
                   peak_velocity, peak_momentum_frame, release_frame,
                   momentum_all, all_frames):
    """Summary table of all key run-up numbers."""
    ax.set_title("Run-up Metrics Summary",
                 fontsize=13, color="white", fontweight="bold")
    ax.axis("off")

    # Momentum at release
    mom_at_release = "N/A"
    if release_frame is not None and release_frame in all_frames:
        idx            = all_frames.index(release_frame)
        mom_at_release = f"{momentum_all[idx]:.1f}"

    # Peak momentum vs release flag
    if peak_momentum_frame is not None and release_frame is not None:
        if peak_momentum_frame > release_frame:
            pm_note = f"{peak_momentum_frame} (post-release)"
        else:
            pm_note = f"{peak_momentum_frame} (pre-release WARNING)"
    else:
        pm_note = str(peak_momentum_frame) if peak_momentum_frame else "N/A"

    rows = [
        ["Metric",                   "Value"],
        ["Approach Angle",           f"{approach_angle:.1f}d"],
        ["Backfoot Contact Frame",   str(backfoot_frame) if backfoot_frame else "N/A"],
        ["Frontfoot Contact Frame",  str(frontfoot_frame) if frontfoot_frame else "N/A"],
        ["Peak Velocity",            f"{peak_velocity:.1f} px/frame"],
        ["Average Gate Width",       f"{average_gate:.1f} px"],
        ["Shoulder Width",           f"{shoulder_width:.1f} px"],
        ["Gate / Shoulder Ratio",    f"{average_gate/shoulder_width:.2f}" if shoulder_width > 0 else "N/A"],
        ["Stride Count",             str(stride_count)],
        ["Peak Momentum Frame",      pm_note],
        ["Momentum at Release",      mom_at_release],
    ]

    table = ax.table(cellText=rows, cellLoc="left",
                     loc="center", colWidths=[0.55, 0.45],
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for j in range(2):
        table[(0, j)].set_facecolor("#1f77b4")
        table[(0, j)].set_text_props(weight="bold", color="white")

    for i in range(1, len(rows)):
        for j in range(2):
            table[(i, j)].set_facecolor("#2a2a2a" if i % 2 == 0 else "#1a1a1a")
            table[(i, j)].set_text_props(color="white")
