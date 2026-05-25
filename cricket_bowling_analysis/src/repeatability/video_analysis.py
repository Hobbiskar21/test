"""Video artifact extraction for repeatability training.

This path creates the frame CSV and pose video needed before the 7-phase
repeatability pipeline runs.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.biomechanics.angle_calculator import compute_angles_with_view_priority
from src.biomechanics.release_detector import find_release_frame
from src.biomechanics.velocity_estimator import compute_all_velocities
from src.biomechanics.bowling_style_detector import get_camera_angle_type, validate_camera_angle
from src.ingestion.frame_extractor import extract_single_camera
from src.pose import SmoothedLandmark, detect_pose_sequence as detect_pose_yolo, smooth_coco_keypoints_sequence
from src.utils.video_utils import get_video_info, write_video
from src.visualization.skeleton_drawer import draw_skeleton


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _resolve_output_root(output_dir: str = None) -> str:
    if output_dir is None:
        return str(Path(__file__).resolve().parents[4] / "outputs")
    path = Path(output_dir)
    if path.is_absolute():
        return str(path)
    return str(Path(__file__).resolve().parents[4] / path)


def _safe_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    if np.isnan(value) or np.isinf(value):
        return ""
    return value


def _point(kpts, idx: int):
    if kpts is None or idx >= len(kpts):
        return None
    x, y = kpts[idx]
    if np.isnan(x) or np.isnan(y):
        return None
    return float(x), float(y)


def _midpoint(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0


def _draw_wrist_trail(frame, kpts_sequence, frame_idx, wrist_idx=10, history=10):
    points = []
    start = max(0, frame_idx - history + 1)
    for idx in range(start, frame_idx + 1):
        if idx >= len(kpts_sequence):
            continue
        point = _point(kpts_sequence[idx], wrist_idx)
        if point is not None:
            points.append((int(point[0]), int(point[1])))

    if len(points) >= 2:
        import cv2
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (255, 255, 255), 2, cv2.LINE_AA)

    if points:
        import cv2
        cv2.circle(frame, points[-1], 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, points[-1], 2, (255, 255, 255), -1, cv2.LINE_AA)

    return frame


def _save_repeatability_frame_csv(
    output_dir,
    session_id,
    smoothed_keypoints,
    angle_seq,
    velocities,
    release_frame,
) -> str:
    """Save frame data with the exact aliases consumed by repeatability."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{session_id}_frame_data.csv")
    wrist_velocity_seq = velocities.get("wrist_velocity_seq", [])
    hip_velocity_seq = velocities.get("hip_velocity_seq", [])

    rows = []
    max_frames = max(len(smoothed_keypoints), len(angle_seq), len(wrist_velocity_seq), len(hip_velocity_seq))
    for frame_idx in range(max_frames):
        kpts = smoothed_keypoints[frame_idx] if frame_idx < len(smoothed_keypoints) else None
        angles = angle_seq[frame_idx] if frame_idx < len(angle_seq) else {}

        nose = _point(kpts, 0)
        left_wrist = _point(kpts, 9)
        right_wrist = _point(kpts, 10)
        left_hip = _point(kpts, 11)
        right_hip = _point(kpts, 12)
        left_ankle = _point(kpts, 15)
        right_ankle = _point(kpts, 16)
        hip_center = _midpoint(left_hip, right_hip)
        body_center = _midpoint(nose, hip_center)

        row = {
            "frame_id": frame_idx,
            "frame": frame_idx,
            "session_id": session_id,
            "delivery_id": session_id,
            "bowler_id": "unknown_bowler",
            "video_id": session_id,
            "phase": "repeatability_pending",
            "phase_id": -1,
            "normalized_phase_time": 0.0,
            "is_release_frame": int(release_frame is not None and frame_idx == release_frame),
            "wrist_speed": _safe_value(wrist_velocity_seq[frame_idx]) if frame_idx < len(wrist_velocity_seq) else "",
            "hip_speed": _safe_value(hip_velocity_seq[frame_idx]) if frame_idx < len(hip_velocity_seq) else "",
            "bowling_wrist_x": _safe_value(right_wrist[0]) if right_wrist else "",
            "bowling_wrist_y": _safe_value(right_wrist[1]) if right_wrist else "",
            "front_ankle_x": _safe_value(left_ankle[0]) if left_ankle else "",
            "front_ankle_y": _safe_value(left_ankle[1]) if left_ankle else "",
            "back_ankle_x": _safe_value(right_ankle[0]) if right_ankle else "",
            "back_ankle_y": _safe_value(right_ankle[1]) if right_ankle else "",
            "hip_center_x": _safe_value(hip_center[0]) if hip_center else "",
            "hip_center_y": _safe_value(hip_center[1]) if hip_center else "",
            "head_x": _safe_value(nose[0]) if nose else "",
            "head_y": _safe_value(nose[1]) if nose else "",
            "body_center_x": _safe_value(body_center[0]) if body_center else "",
            "body_center_y": _safe_value(body_center[1]) if body_center else "",
        }

        for name, value in angles.items():
            row[name] = _safe_value(value)

        row["trunk_lean_angle"] = row.get("trunk_lean", "")
        row["bowling_elbow_angle"] = row.get("elbow_angle", "")
        row["bowling_arm_angle"] = row.get("shoulder_angle", "")
        row["front_leg_angle"] = row.get("front_knee_angle", "")
        row["back_leg_angle"] = row.get("back_knee_angle", "")
        row["front_knee_angle_velocity"] = ""
        row["arm_angle_velocity"] = ""
        row["trunk_lean_velocity"] = ""
        row["head_speed"] = ""
        row["front_ankle_speed"] = ""
        if left_ankle and right_ankle:
            row["stride_length_proxy"] = float(np.linalg.norm(np.array(left_ankle) - np.array(right_ankle)))
        else:
            row["stride_length_proxy"] = ""
        row["release_height_proxy"] = row["bowling_wrist_y"]

        rows.append(row)

    df = pd.DataFrame(rows)
    for source, target in (
        ("front_knee_angle", "front_knee_angle_velocity"),
        ("bowling_arm_angle", "arm_angle_velocity"),
        ("trunk_lean_angle", "trunk_lean_velocity"),
    ):
        if source in df.columns:
            values = pd.to_numeric(df[source], errors="coerce").ffill().bfill()
            df[target] = values.diff().fillna(0.0)

    if {"head_x", "head_y"}.issubset(df.columns):
        head_x = pd.to_numeric(df["head_x"], errors="coerce").ffill().bfill()
        head_y = pd.to_numeric(df["head_y"], errors="coerce").ffill().bfill()
        df["head_speed"] = np.sqrt(head_x.diff().fillna(0.0) ** 2 + head_y.diff().fillna(0.0) ** 2)

    if {"front_ankle_x", "front_ankle_y"}.issubset(df.columns):
        ankle_x = pd.to_numeric(df["front_ankle_x"], errors="coerce").ffill().bfill()
        ankle_y = pd.to_numeric(df["front_ankle_y"], errors="coerce").ffill().bfill()
        df["front_ankle_speed"] = np.sqrt(ankle_x.diff().fillna(0.0) ** 2 + ankle_y.diff().fillna(0.0) ** 2)

    df.to_csv(csv_path, index=False)
    print(f"[REPEATABILITY VIDEO] Frame-by-frame CSV saved: {csv_path}")
    return csv_path


def _save_keypoints_csv(output_dir, session_id, raw_keypoints, smoothed_keypoints) -> str:
    """Save raw and smoothed COCO keypoints frame by frame."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{session_id}_keypoints.csv")
    max_frames = max(len(raw_keypoints), len(smoothed_keypoints))
    rows = []

    for frame_idx in range(max_frames):
        row = {
            "session_id": session_id,
            "video_id": session_id,
            "frame_id": frame_idx,
            "frame": frame_idx,
        }
        for prefix, sequence in (("raw", raw_keypoints), ("smoothed", smoothed_keypoints)):
            kpts = sequence[frame_idx] if frame_idx < len(sequence) else None
            for kpt_idx, name in enumerate(COCO_KEYPOINT_NAMES):
                if kpts is not None and kpt_idx < len(kpts):
                    row[f"{prefix}_{name}_x"] = _safe_value(kpts[kpt_idx][0])
                    row[f"{prefix}_{name}_y"] = _safe_value(kpts[kpt_idx][1])
                else:
                    row[f"{prefix}_{name}_x"] = ""
                    row[f"{prefix}_{name}_y"] = ""
        rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[REPEATABILITY VIDEO] Keypoints CSV saved: {csv_path}")
    return csv_path


def run_repeatability_video_analysis(video_path: str, output_dir: str = None, render_video: bool = True) -> dict:
    """Create repeatability frame/video artifacts without 4-phase segmentation."""
    output_root = _resolve_output_root(output_dir)
    video_output_dir = os.path.join(output_root, "videos")
    artifact_output_dir = os.path.join(output_root, "artifacts")
    if render_video:
        os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(artifact_output_dir, exist_ok=True)

    print(f"\n[REPEATABILITY VIDEO] Loading: {video_path}")
    info = get_video_info(video_path)
    w, h = info["width"], info["height"]
    fps = info["fps"]
    frame_count = info["frame_count"]
    print(f"[REPEATABILITY VIDEO] Resolution: {w}x{h} | FPS: {fps} | Frames: {frame_count}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    session_id = f"single_{video_name}"

    print(f"\n[REPEATABILITY VIDEO] Extracting frames...")
    original_frames, _ = extract_single_camera(video_path, offset=0)
    print(f"[REPEATABILITY VIDEO] Extracted {len(original_frames)} frames")

    print(f"\n[REPEATABILITY VIDEO] Detecting pose with YOLO...")
    coco_landmarks_raw, _ = detect_pose_yolo(
        original_frames,
        w,
        h,
        pose_model="yolov8m-pose.pt",
        imgsz=960,
        max_jump_px=260.0,
        device="cuda",
    )

    print(f"[REPEATABILITY VIDEO] Smoothing COCO keypoints...")
    coco_landmarks_smoothed = smooth_coco_keypoints_sequence(coco_landmarks_raw)

    landmarks_smoothed = []
    for coco_kpts in coco_landmarks_smoothed:
        if coco_kpts is None:
            landmarks_smoothed.append(None)
            continue
        landmarks = []
        for i in range(17):
            x_px, y_px = coco_kpts[i]
            x_norm = x_px / w if w > 0 else 0
            y_norm = y_px / h if h > 0 else 0
            landmarks.append(SmoothedLandmark(x_norm, y_norm, 0.0, 1.0))
        landmarks_smoothed.append(landmarks)

    print(f"\n[REPEATABILITY VIDEO] Detecting camera angle...")
    _, camera_msg = validate_camera_angle(landmarks_smoothed, w, h)
    print(f"[INFO] {camera_msg}")
    camera_angle_type = get_camera_angle_type(landmarks_smoothed, w, h)
    print(f"[INFO] Detected camera angle: {camera_angle_type}")

    print(f"\n[REPEATABILITY VIDEO] Computing velocities...")
    velocities = compute_all_velocities(landmarks_smoothed, fps, w, h)

    wrist_positions = []
    for lm in landmarks_smoothed:
        if lm is not None and lm[16].visibility > 0.5:
            wrist_positions.append((lm[16].x * w, lm[16].y * h))
        else:
            wrist_positions.append(None)

    print(f"\n[REPEATABILITY VIDEO] Computing angles...")
    angle_seq = []
    angle_metadata_seq = []
    for lm in landmarks_smoothed:
        if lm is not None:
            angles, metadata = compute_angles_with_view_priority(lm, w, h, camera_view=camera_angle_type)
            angle_seq.append(angles)
            angle_metadata_seq.append(metadata)
        else:
            angle_seq.append({})
            angle_metadata_seq.append({})

    release = find_release_frame(
        velocities["wrist_velocity_seq"],
        wrist_positions=wrist_positions,
        phase_map=None,
    )
    print(f"[REPEATABILITY VIDEO] Approx release frame for CSV marker: {release}")

    phase_map = {i: "repeatability_pending" for i in range(len(original_frames))}

    print(f"\n[REPEATABILITY VIDEO] Saving frame CSV...")
    frame_csv = _save_repeatability_frame_csv(
        output_dir=artifact_output_dir,
        session_id=session_id,
        smoothed_keypoints=coco_landmarks_smoothed,
        angle_seq=angle_seq,
        velocities=velocities,
        release_frame=release,
    )
    keypoints_csv = _save_keypoints_csv(
        output_dir=artifact_output_dir,
        session_id=session_id,
        raw_keypoints=coco_landmarks_raw,
        smoothed_keypoints=coco_landmarks_smoothed,
    )
    marker_path = os.path.join(artifact_output_dir, f"{session_id}_repeatability_only.txt")
    Path(marker_path).write_text("repeatability_video_analysis=1\n")

    out_path = os.path.join(video_output_dir, f"{session_id}_analysis.mp4")
    if render_video:
        print(f"\n[REPEATABILITY VIDEO] Rendering pose video...")
        annotated = []
        for i, frame in enumerate(original_frames):
            lm = landmarks_smoothed[i]
            render_kpts = coco_landmarks_raw[i] if i < len(coco_landmarks_raw) else None
            angles = angle_seq[i]
            angle_metadata = angle_metadata_seq[i]

            angle_annotations = {}
            for key, value in angles.items():
                if value is None:
                    continue
                try:
                    val_float = float(value)
                    if not np.isnan(val_float) and not np.isinf(val_float) and -360 <= val_float <= 360:
                        angle_annotations[key] = val_float
                except (TypeError, ValueError, OverflowError):
                    pass

            out = draw_skeleton(
                frame,
                render_kpts if render_kpts is not None else lm,
                w,
                h,
                angle_annotations={},
                phase_label="",
                bowling_style=None,
                velocities=None,
                angle_metadata=None,
                frame_number=i,
            )
            out = _draw_wrist_trail(out, coco_landmarks_raw, i, wrist_idx=10, history=10)
            annotated.append(out)

        video_ok = write_video(annotated, out_path, fps, slowdown=0.25)
        if video_ok:
            print(f"[REPEATABILITY VIDEO] Video saved: {out_path}")
        else:
            print(f"[ERROR] Video rendering failed: {out_path}")
    else:
        out_path = None

    return {
        "session_id": session_id,
        "frame_csv": frame_csv,
        "keypoints_csv": keypoints_csv,
        "analysis_video": out_path,
        "repeatability_marker": marker_path,
        "release_frame": release,
    }
