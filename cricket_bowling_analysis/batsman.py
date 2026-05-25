"""Export clean batsman test artifacts for one normal single video.

Output folder:
    outputs/test_output/batsman/video
    outputs/test_output/batsman/keypoints_csv
    outputs/test_output/batsman/metrics_csv
    outputs/test_output/batsman/phases_csv
    outputs/test_output/batsman/analysis_csv

Default input folder:
    input_videos/test/batsman
"""

import argparse
import os
import shutil
import stat
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.batting.feature_extraction import BattingFeatureExtractor
from src.batting.phase_detection import BattingPhaseDetector
from src.batting.smoothening import exponential_smooth_coco_keypoints_sequence
from src.ingestion.frame_extractor import extract_single_camera
from src.pose import detect_pose_sequence as detect_pose_yolo
from src.repeatability.video_analysis import COCO_KEYPOINT_NAMES
from src.utils.video_utils import get_video_info, write_video
from src.visualization.coco_skeleton_drawer import draw_coco_skeleton


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_test_video_dir() -> Path:
    return workspace_root() / "input_videos" / "test" / "batsman"


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return workspace_root() / path


def ensure_input_folder(video_dir: Path) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)


def find_videos(video_dir: Path) -> list[Path]:
    ensure_input_folder(video_dir)
    return sorted(path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)


def choose_video(video_dir: Path) -> Path:
    videos = find_videos(video_dir)
    if not videos:
        raise FileNotFoundError(
            f"No videos found in: {video_dir}\n"
            f"Put your batsman test videos in this folder, then run this file again."
        )

    print("\nSelect batsman test video:")
    for idx, video in enumerate(videos, start=1):
        print(f"  {idx}. {video.name}")

    while True:
        choice = input("Enter number: ").strip()
        try:
            selected = int(choice)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= selected <= len(videos):
            return videos[selected - 1]
        print(f"Please choose between 1 and {len(videos)}.")


def _on_rm_error(func, path, exc_info):
    exc_type, exc_value, _ = exc_info
    if exc_type is PermissionError:
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            raise
    else:
        raise


def clear_output_folder(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir, onerror=_on_rm_error)
    (output_dir / "video").mkdir(parents=True, exist_ok=True)
    (output_dir / "keypoints_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "phases_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_csv").mkdir(parents=True, exist_ok=True)


def _safe_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return ""
    if pd.isna(value):
        return ""
    return value


def save_keypoints_csv(output_path: Path, video_id: str, raw_keypoints, smoothed_keypoints) -> Path:
    rows = []
    max_frames = max(len(raw_keypoints), len(smoothed_keypoints))
    for frame_idx in range(max_frames):
        row = {
            "video_id": video_id,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[BATSMAN TEST] Saved keypoints CSV: {output_path}")
    return output_path


def _safe_numeric(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    if pd.isna(value):
        return np.nan
    return value


def load_batting_keypoints_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    point_columns = [f"x{i}" for i in range(17)] + [f"y{i}" for i in range(17)]
    if all(col in df.columns for col in point_columns):
        keypoints_data = df[point_columns].astype(float).fillna(0).values
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        keypoints_data = df[numeric_cols[:34]].astype(float).fillna(0).values
    return keypoints_data


def flatten_keypoints_sequence(keypoints_sequence: list) -> np.ndarray:
    rows = []
    for frame_keypoints in keypoints_sequence:
        row = np.full(34, np.nan, dtype=float)
        if frame_keypoints is not None and len(frame_keypoints) >= 17:
            for kpt_idx in range(17):
                row[kpt_idx * 2] = _safe_numeric(frame_keypoints[kpt_idx][0])
                row[kpt_idx * 2 + 1] = _safe_numeric(frame_keypoints[kpt_idx][1])
        rows.append(row)
    return np.vstack(rows) if rows else np.empty((0, 34), dtype=float)


def save_features_csv(output_path: Path, keypoints_data: np.ndarray) -> pd.DataFrame:
    extractor = BattingFeatureExtractor(right_handed=True)
    feature_rows = []
    features_list = extractor.extract_features_batch(keypoints_data)
    for frame_id, features in enumerate(features_list):
        feature_rows.append({
            "frame_id": frame_id,
            "weight_transfer": features.get("weight_transfer", np.nan),
            "front_knee_angle": features.get("front_knee_angle", np.nan),
            "head_stability_index": features.get("head_stability_index", np.nan),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(feature_rows)
    df.to_csv(output_path, index=False)
    print(f"[BATSMAN TEST] Saved batting features CSV: {output_path}")
    return df


def save_metrics_csv(output_path: Path, features_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = features_df[["frame_id", "weight_transfer", "front_knee_angle", "head_stability_index"]].copy()
    metrics_df["weight_transfer_how_calculated"] = (
        "Right-handed batter assumption: front side is left and back side is right. "
        "Computed as front_hip_x / (back_hip_x + front_hip_x), clipped to 0-1."
    )
    metrics_df["front_knee_angle_how_calculated"] = (
        "Right-handed batter assumption: front knee is left knee. "
        "Angle in degrees at the front knee using front hip, front knee, and front ankle keypoints."
    )
    metrics_df["head_stability_index_how_calculated"] = (
        "Uses nose x/y positions across the full video. "
        "Computed as 1 - ((variance_x + variance_y) / head_position_range_squared), clipped to 0-1."
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"[BATSMAN TEST] Saved batting metrics CSV: {output_path}")
    return metrics_df


def save_phases_csv(output_path: Path, keypoints_data: np.ndarray) -> pd.DataFrame:
    detector = BattingPhaseDetector(right_handed=True)
    phases = detector.detect_phases(keypoints_data)
    rows = []
    for frame_id, phase in enumerate(phases):
        rows.append({
            "frame_id": frame_id,
            "phase": int(phase),
            "phase_name": detector.get_phase_name(int(phase)),
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[BATSMAN TEST] Saved batting phases CSV: {output_path}")
    return df


def save_batting_analysis_csv(output_path: Path, features_df: pd.DataFrame, phases_df: pd.DataFrame) -> pd.DataFrame:
    merged = features_df.merge(phases_df, on="frame_id", how="outer").sort_values("frame_id").reset_index(drop=True)
    merged["shot_id"] = 0
    columns = ["shot_id", "frame_id", "phase", "phase_name", "weight_transfer", "front_knee_angle", "head_stability_index"]
    merged = merged[[col for col in columns if col in merged.columns]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"[BATSMAN TEST] Saved batting analysis CSV: {output_path}")
    return merged


def draw_phase_angle_overlay(frame, keypoints, phase_name: str, front_knee_angle: float, show_angle: bool = True, right_handed=True):
    out = draw_coco_skeleton(frame, keypoints, radius=4, thickness=2)
    height, width = out.shape[:2]
    banner_h = min(68, max(48, int(height * 0.08)))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_h), (248, 249, 250), -1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)
    cv2.line(out, (0, banner_h), (width, banner_h), (35, 35, 35), 2)

    title = f"PHASE: {phase_name.upper()}"
    y_text = max(30, int(banner_h * 0.55))
    cv2.putText(out, title, (18, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.putText(out, title, (18, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 0), 1, cv2.LINE_AA)

    if show_angle:
        angle_text = f"Front knee angle: {front_knee_angle:.1f} deg"
        cv2.putText(out, angle_text, (18, y_text + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)

    if show_angle and keypoints is not None and len(keypoints) >= 17:
        try:
            if right_handed:
                hip_idx, knee_idx, ankle_idx = 11, 13, 15
            else:
                hip_idx, knee_idx, ankle_idx = 12, 14, 16

            hip = keypoints[hip_idx]
            knee = keypoints[knee_idx]
            ankle = keypoints[ankle_idx]
            hip_pt = (int(round(hip[0])), int(round(hip[1])))
            knee_pt = (int(round(knee[0])), int(round(knee[1])))
            ankle_pt = (int(round(ankle[0])), int(round(ankle[1])))

            if all(np.isfinite([hip_pt[0], hip_pt[1], knee_pt[0], knee_pt[1], ankle_pt[0], ankle_pt[1]])):
                cv2.line(out, hip_pt, knee_pt, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(out, knee_pt, ankle_pt, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(out, knee_pt, 6, (0, 0, 255), -1, cv2.LINE_AA)

                vec1 = np.array(hip_pt, dtype=float) - np.array(knee_pt, dtype=float)
                vec2 = np.array(ankle_pt, dtype=float) - np.array(knee_pt, dtype=float)
                angle_between = float(front_knee_angle) if np.isfinite(front_knee_angle) else 0.0
                if angle_between > 0 and np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    start_angle = np.degrees(np.arctan2(vec1[1], vec1[0]))
                    extent = min(180.0, abs(angle_between))
                    radius = max(30, min(width, height) // 10)
                    cv2.ellipse(out, knee_pt, (radius, radius), float(start_angle), 0.0, float(extent), (0, 0, 255), 3, cv2.LINE_AA)
        except Exception:
            pass

    return out


def render_phase_angle_video(output_path: Path, frames, raw_keypoints, phases_df: pd.DataFrame, features_df: pd.DataFrame, fps: float) -> Path:
    if fps <= 0:
        fps = 25.0
    slowdown = 0.2
    output_fps = max(float(fps) * slowdown, 1.0)
    pause_frames = max(1, int(round(output_fps * 1.0)))

    phase_names = phases_df.set_index('frame_id')['phase_name'].to_dict()
    angles = features_df.set_index('frame_id')['front_knee_angle'].to_dict()
    phase_values = [int(x) for x in phases_df['phase'].tolist()]
    phase_changes = {0}
    for previous, current, frame_id in zip(phase_values, phase_values[1:], phases_df['frame_id'].tolist()[1:]):
        if current != previous:
            phase_changes.add(int(frame_id))

    annotated_frames = []
    for frame_id, frame in enumerate(frames):
        phase_name = phase_names.get(frame_id, 'Unknown')
        angle = float(angles.get(frame_id, 0.0) or 0.0)
        keypoints = raw_keypoints[frame_id] if frame_id < len(raw_keypoints) else None
        show_angle = frame_id in phase_changes
        labelled = draw_phase_angle_overlay(frame, keypoints, phase_name, angle, show_angle=show_angle)
        annotated_frames.append(labelled)
        if frame_id in phase_changes:
            for _ in range(pause_frames):
                annotated_frames.append(labelled.copy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = write_video(annotated_frames, str(output_path), fps, slowdown=slowdown)
    if not ok:
        raise RuntimeError(f"Could not write phase/angle video: {output_path}")
    print(f"[BATSMAN TEST] Saved phase/angle video: {output_path}")
    return output_path


def export_video(video_path: Path, output_dir: Path, smoothing_alpha: float = 0.15) -> dict[str, Path]:
    clear_output_folder(output_dir)
    print(f"[BATSMAN TEST] Processing video: {video_path}")

    info = get_video_info(str(video_path))
    frames, _ = extract_single_camera(str(video_path), offset=0)
    raw_keypoints, _ = detect_pose_yolo(
        frames,
        info["width"],
        info["height"],
        pose_model="yolov8m-pose.pt",
        imgsz=960,
        max_jump_px=260.0,
        device=None,
    )
    smoothed_keypoints = exponential_smooth_coco_keypoints_sequence(raw_keypoints, alpha=smoothing_alpha)

    keypoints_csv = save_keypoints_csv(
        output_dir / "keypoints_csv" / f"{video_path.stem}_keypoints.csv",
        video_path.stem,
        raw_keypoints,
        smoothed_keypoints,
    )
    return {
        "keypoints_csv": keypoints_csv,
        "frames": frames,
        "raw_keypoints": raw_keypoints,
        "smoothed_keypoints": smoothed_keypoints,
        "fps": info["fps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clean batsman test output for one normal video.")
    parser.add_argument("--video", default=None, help="Optional path to one batsman video.")
    parser.add_argument("--video_dir", default=None, help="Folder to choose a video from. Defaults to input_videos/test/batsman.")
    parser.add_argument("--output_dir", default="outputs/test_output/batsman", help="Clean batsman output folder.")
    parser.add_argument("--smoothing_alpha", type=float, default=0.15, help="Exponential smoothing alpha for the batting video overlay.")
    args = parser.parse_args()

    if args.video:
        video_path = Path(args.video)
    else:
        video_dir = Path(args.video_dir) if args.video_dir else default_test_video_dir()
        ensure_input_folder(video_dir)
        print(f"[BATSMAN TEST] Input video folder: {video_dir}")
        video_path = choose_video(video_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = resolve_output_dir(args.output_dir)
    results = export_video(video_path.resolve(), output_dir, smoothing_alpha=args.smoothing_alpha)

    keypoints_data = flatten_keypoints_sequence(results["raw_keypoints"])
    features_df = save_features_csv(output_dir / "metrics_csv" / f"{video_path.stem}_batting_features.csv", keypoints_data)
    metrics_df = save_metrics_csv(output_dir / "metrics_csv" / f"{video_path.stem}_batting_metrics.csv", features_df)
    phases_df = save_phases_csv(output_dir / "phases_csv" / f"{video_path.stem}_batting_phases.csv", keypoints_data)
    analysis_df = save_batting_analysis_csv(output_dir / "analysis_csv" / f"{video_path.stem}_batting_analysis.csv", features_df, phases_df)
    phase_video = render_phase_angle_video(
        output_dir / "video" / f"{video_path.stem}_phase_angles.mp4",
        results["frames"],
        results["smoothed_keypoints"],
        phases_df,
        features_df,
        results["fps"],
    )

    print("\n[BATSMAN TEST] Done.")
    print(f"  Final video: {phase_video}")
    print(f"  Keypoints CSV: {results['keypoints_csv']}")
    print(f"  Features CSV: {output_dir / 'metrics_csv' / f'{video_path.stem}_batting_features.csv'}")
    print(f"  Metrics CSV: {output_dir / 'metrics_csv' / f'{video_path.stem}_batting_metrics.csv'}")
    print(f"  Phases CSV: {output_dir / 'phases_csv' / f'{video_path.stem}_batting_phases.csv'}")
    print(f"  Analysis CSV: {output_dir / 'analysis_csv' / f'{video_path.stem}_batting_analysis.csv'}")


if __name__ == "__main__":
    main()
