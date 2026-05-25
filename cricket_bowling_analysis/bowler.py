"""Export clean repeatability test artifacts for side-on bowling videos.

Output folder:
    outputs/test_output/bowling/video
    outputs/test_output/bowling/phase_label_csv
    outputs/test_output/bowling/keypoints_csv
    outputs/test_output/bowling/features_csv
    outputs/test_output/bowling/metrics_csv

Default input folder:
    input_videos/test/bowling
"""

import argparse
import os
import shutil
import stat
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.repeatability.config import PHASE_DISPLAY_NAMES, PHASE_NAMES
from src.repeatability.phase_segmenter import assign_phases_to_frames
from src.repeatability.repeatability_input_builder import build_repeatability_input
from src.repeatability.sideon_feature_selector import select_sideon_features
from src.repeatability.sideon_phase_detector import detect_sideon_phases
from src.repeatability.video_analysis import run_repeatability_video_analysis


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def phase_for_frame(phases_df: pd.DataFrame, frame_id: int) -> str:
    matches = phases_df[(phases_df["start_frame"] <= frame_id) & (phases_df["end_frame"] >= frame_id)]
    if matches.empty:
        return "unassigned"
    return str(matches.iloc[0]["phase"])


def _phase_index(phase: str) -> int:
    try:
        return PHASE_NAMES.index(phase) + 1
    except ValueError:
        return 0


def draw_phase_label(frame, phase: str, frame_id: int):
    out = frame.copy()
    height, width = out.shape[:2]
    label = PHASE_DISPLAY_NAMES.get(phase, phase)
    idx = _phase_index(phase)
    title = f"PHASE {idx}/7" if idx else "PHASE"
    subtitle = label.upper() if idx else "UNASSIGNED"

    banner_h = min(150, max(105, int(height * 0.16)))
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_h), (245, 246, 248), -1)
    out = cv2.addWeighted(overlay, 0.86, out, 0.14, 0)
    cv2.line(out, (0, banner_h), (width, banner_h), (25, 25, 25), 3)

    cv2.putText(out, title, (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (15, 15, 15), 5, cv2.LINE_AA)
    cv2.putText(out, title, (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, subtitle, (28, 103), cv2.FONT_HERSHEY_SIMPLEX, 1.55, (15, 15, 15), 6, cv2.LINE_AA)
    cv2.putText(out, subtitle, (28, 103), cv2.FONT_HERSHEY_SIMPLEX, 1.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, f"Frame {frame_id}", (30, banner_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2, cv2.LINE_AA)
    return out


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_test_video_dir() -> Path:
    return workspace_root() / "input_videos" / "test" / "bowling"


def ensure_input_folder(video_dir: Path) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)


def resolve_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return workspace_root() / path


def find_videos(video_dir: Path) -> list[Path]:
    ensure_input_folder(video_dir)
    return sorted(path for path in video_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)


def choose_video(video_dir: Path) -> Path:
    videos = find_videos(video_dir)
    if not videos:
        raise FileNotFoundError(
            f"No videos found in: {video_dir}\n"
            f"Put your bowling test videos in this folder, then run this file again."
        )

    print("\nSelect repeatability test video:")
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
    (output_dir / "phase_label_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "keypoints_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "features_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_csv").mkdir(parents=True, exist_ok=True)


def ensure_output_folders(output_dir: Path) -> None:
    (output_dir / "video").mkdir(parents=True, exist_ok=True)
    (output_dir / "phase_label_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "keypoints_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "features_csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_csv").mkdir(parents=True, exist_ok=True)


def copy_file(source: Path, destination: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Expected file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        index = 2
        original_destination = destination
        while destination.exists():
            destination = original_destination.with_name(
                f"{original_destination.stem}_{index}{original_destination.suffix}"
            )
            index += 1
    shutil.copy2(source, destination)
    return destination


def _nearest_row(df: pd.DataFrame, frame_id: int) -> pd.Series:
    frame_numbers = pd.to_numeric(df["frame_id"], errors="coerce")
    idx = (frame_numbers - frame_id).abs().idxmin()
    return df.loc[idx]


def _event_frame(phases_df: pd.DataFrame, phase: str, fallback: str) -> int:
    phase_rows = phases_df[phases_df["phase"] == phase]
    if not phase_rows.empty and "event_frame" in phase_rows.columns:
        event = phase_rows.iloc[0].get("event_frame")
        if pd.notna(event):
            return int(event)
    if phase_rows.empty:
        raise ValueError(f"Phase not found in phase detector output: {phase}")
    return int(phase_rows.iloc[0][fallback])


def _phase_at_frame(phase_label_df: pd.DataFrame, frame_id: int) -> str:
    row = _nearest_row(phase_label_df, frame_id)
    return str(row.get("phase", "unassigned"))


def _value_at_frame(df: pd.DataFrame, frame_id: int, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    value = _nearest_row(df, frame_id).get(column)
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def add_metric_columns_to_features(features_path: Path, repeat_input_path: Path) -> None:
    features_df = pd.read_csv(features_path)
    input_df = pd.read_csv(repeat_input_path)
    extra_columns = ["lateral_flexion", "trunk_flexion", "hip_shoulder_sep"]
    available = [col for col in extra_columns if col in input_df.columns and col not in features_df.columns]
    if not available:
        return
    merged = features_df.merge(input_df[["frame_id"] + available], on="frame_id", how="left")
    merged.to_csv(features_path, index=False)


def build_bowling_metrics_csv(
    features_path: Path,
    phase_label_path: Path,
    phases_path: Path,
    output_path: Path,
) -> tuple[Path, dict[str, int]]:
    features_df = pd.read_csv(features_path)
    phase_label_df = pd.read_csv(phase_label_path)
    phases_df = pd.read_csv(phases_path)

    ffc_frame = _event_frame(phases_df, "ffc_window", "start_frame")
    release_frame = _event_frame(phases_df, "ffc_to_release", "end_frame")

    rows = [
        {
            "metric": "Front knee angle at Front Foot Contact",
            "value": round(_value_at_frame(features_df, ffc_frame, "front_knee_angle"), 3),
            "unit": "degrees",
            "frame_id": ffc_frame,
            "phase": _phase_at_frame(phase_label_df, ffc_frame),
            "source_column": "front_knee_angle",
            "how_taken": "Used the FFC event_frame from the detected ffc_window phase, then read front_knee_angle from the features CSV at the nearest frame.",
        },
        {
            "metric": "Delivery stride length",
            "value": round(_value_at_frame(features_df, ffc_frame, "stride_length_proxy"), 3),
            "unit": "pixels",
            "frame_id": ffc_frame,
            "phase": _phase_at_frame(phase_label_df, ffc_frame),
            "source_column": "stride_length_proxy",
            "how_taken": "Used the same FFC frame and read stride_length_proxy from the features CSV; it is the pixel distance between front and back ankle keypoints.",
        },
        {
            "metric": "Trunk lateral flexion at release",
            "value": round(_value_at_frame(features_df, release_frame, "lateral_flexion"), 3),
            "unit": "degrees",
            "frame_id": release_frame,
            "phase": _phase_at_frame(phase_label_df, release_frame),
            "source_column": "lateral_flexion",
            "how_taken": "Used the release event_frame from the detected ffc_to_release phase, then read lateral_flexion from the features CSV at the nearest frame.",
        },
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[TEST OUTPUT] Saved bowling metrics CSV: {output_path}")
    return output_path, {"ffc": ffc_frame, "release": release_frame}


def _point(row: pd.Series, keypoint_name: str, prefix: str = "smoothed"):
    x = pd.to_numeric(pd.Series([row.get(f"{prefix}_{keypoint_name}_x")]), errors="coerce").iloc[0]
    y = pd.to_numeric(pd.Series([row.get(f"{prefix}_{keypoint_name}_y")]), errors="coerce").iloc[0]
    if pd.isna(x) or pd.isna(y):
        return None
    return np.array([float(x), float(y)])


def _midpoint(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def _draw_angle_arc(frame, a, b, c, label: str, value: float, color=(0, 190, 255)):
    if a is None or b is None or c is None or pd.isna(value):
        return frame
    out = frame.copy()
    pa, pb, pc = tuple(a.astype(int)), tuple(b.astype(int)), tuple(c.astype(int))
    cv2.line(out, pb, pa, color, 4, cv2.LINE_AA)
    cv2.line(out, pb, pc, color, 4, cv2.LINE_AA)
    cv2.circle(out, pb, 7, color, -1, cv2.LINE_AA)

    angle1 = np.degrees(np.arctan2(a[1] - b[1], a[0] - b[0]))
    angle2 = np.degrees(np.arctan2(c[1] - b[1], c[0] - b[0]))
    start, end = sorted([(angle1 + 360) % 360, (angle2 + 360) % 360])
    if end - start > 180:
        start, end = end, start + 360
    radius = max(22, int(min(np.linalg.norm(a - b), np.linalg.norm(c - b)) * 0.35))
    cv2.ellipse(out, pb, (radius, radius), 0, start, end, color, 3, cv2.LINE_AA)

    text = f"{label}: {value:.1f} deg"
    text_pos = (max(12, min(out.shape[1] - 360, pb[0] + 14)), max(42, pb[1] - 16))
    cv2.putText(out, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return out


def _draw_metric_arcs(frame, frame_id: int, keypoints_df: pd.DataFrame, metrics_df: pd.DataFrame, metric_frames: dict[str, int]):
    out = frame
    keypoint_row = _nearest_row(keypoints_df, frame_id)
    metrics_by_name = {row["metric"]: row for _, row in metrics_df.iterrows()}

    if abs(frame_id - metric_frames["ffc"]) <= 1:
        metric = metrics_by_name["Front knee angle at Front Foot Contact"]
        hip = _point(keypoint_row, "left_hip")
        knee = _point(keypoint_row, "left_knee")
        ankle = _point(keypoint_row, "left_ankle")
        out = _draw_angle_arc(out, hip, knee, ankle, "Front knee", float(metric["value"]), (0, 190, 255))

    if abs(frame_id - metric_frames["release"]) <= 1:
        metric = metrics_by_name["Trunk lateral flexion at release"]
        nose = _point(keypoint_row, "nose")
        left_hip = _point(keypoint_row, "left_hip")
        right_hip = _point(keypoint_row, "right_hip")
        right_shoulder = _point(keypoint_row, "right_shoulder")
        mid_hip = _midpoint(left_hip, right_hip)
        out = _draw_angle_arc(out, nose, mid_hip, right_shoulder, "Trunk lateral flexion", float(metric["value"]), (80, 220, 120))

    return out


def render_metric_video(
    analyzed_video_path: Path,
    phases_path: Path,
    keypoints_path: Path,
    metrics_path: Path,
    metric_frames: dict[str, int],
    output_path: Path,
) -> Path:
    phases_df = pd.read_csv(phases_path)
    keypoints_df = pd.read_csv(keypoints_path)
    metrics_df = pd.read_csv(metrics_path)
    pause_event_frames = set(metric_frames.values())
    min_phase_frame = int(phases_df["start_frame"].min())
    max_phase_frame = int(phases_df["end_frame"].max())

    cap = cv2.VideoCapture(str(analyzed_video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open analyzed video: {analyzed_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pause_frames = max(1, int(round(fps * 1.0)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not open writer: {output_path}")

    video_frame_id = 0
    paused_events = set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if total_frames > 1:
            frame_id = int(round(min_phase_frame + (video_frame_id / (total_frames - 1)) * (max_phase_frame - min_phase_frame)))
        else:
            frame_id = video_frame_id
        phase = phase_for_frame(phases_df, frame_id)
        labelled = draw_phase_label(frame, phase, frame_id)
        labelled = _draw_metric_arcs(labelled, frame_id, keypoints_df, metrics_df, metric_frames)
        pause_frame = next((event for event in pause_event_frames if abs(frame_id - event) <= 1), None)
        if pause_frame is not None and pause_frame not in paused_events:
            for _ in range(pause_frames):
                writer.write(labelled)
            paused_events.add(pause_frame)
        writer.write(labelled)
        video_frame_id += 1

    cap.release()
    writer.release()
    print(f"[TEST OUTPUT] Saved metric video: {output_path}")
    return output_path


def export_video(video_path: Path, output_dir: Path, clear_output: bool = False) -> dict[str, Path]:
    if clear_output:
        clear_output_folder(output_dir)
    else:
        ensure_output_folders(output_dir)
    print(f"[TEST OUTPUT] Processing video: {video_path}")

    with tempfile.TemporaryDirectory(prefix="repeatability_test_output_") as temp_dir:
        work_dir = Path(temp_dir)
        analysis_root = work_dir / "video_analysis" / video_path.stem
        repeatability_root = work_dir / "repeatability"

        analysis = run_repeatability_video_analysis(
            str(video_path),
            str(analysis_root),
            render_video=True,
        )

        repeat_input = repeatability_root / "input" / f"{Path(analysis['frame_csv']).stem}_repeatability_input.csv"
        auto_phases = repeatability_root / "auto_phases" / f"{Path(analysis['frame_csv']).stem}_auto_phases.csv"
        auto_events = repeatability_root / "auto_events" / f"{Path(analysis['frame_csv']).stem}_auto_events.csv"
        features = repeatability_root / "features" / f"{video_path.stem}_features.csv"
        feature_report = repeatability_root / "features" / f"{video_path.stem}_features_report.csv"
        phase_labelled = repeatability_root / "phase_labelled" / f"{video_path.stem}_phase_labelled.csv"

        build_repeatability_input(Path(analysis["frame_csv"]), repeat_input)
        detect_sideon_phases(repeat_input, auto_phases, auto_events)
        select_sideon_features(repeat_input, auto_phases, features, feature_report)
        add_metric_columns_to_features(features, repeat_input)
        assign_phases_to_frames(features, auto_phases, phase_labelled)

        metrics_csv = repeatability_root / "metrics" / f"{video_path.stem}_bowling_metrics.csv"
        labelled_video = work_dir / f"{video_path.stem}_7_phase_detection.mp4"
        _, metric_frames = build_bowling_metrics_csv(features, phase_labelled, auto_phases, metrics_csv)
        render_metric_video(
            Path(analysis["analysis_video"]),
            auto_phases,
            Path(analysis["keypoints_csv"]),
            metrics_csv,
            metric_frames,
            labelled_video,
        )

        final_phase_csv = copy_file(
            phase_labelled,
            output_dir / "phase_label_csv" / f"{video_path.stem}_phase_labels.csv",
        )
        final_keypoints_csv = copy_file(
            Path(analysis["keypoints_csv"]),
            output_dir / "keypoints_csv" / f"{video_path.stem}_keypoints.csv",
        )
        final_features_csv = copy_file(
            features,
            output_dir / "features_csv" / f"{video_path.stem}_features.csv",
        )
        final_metrics_csv = copy_file(
            metrics_csv,
            output_dir / "metrics_csv" / f"{video_path.stem}_bowling_metrics.csv",
        )
        final_video = copy_file(
            labelled_video,
            output_dir / "video" / f"{video_path.stem}_7_phase_detection.mp4",
        )

    return {
        "video": final_video,
        "phase_label_csv": final_phase_csv,
        "keypoints_csv": final_keypoints_csv,
        "features_csv": final_features_csv,
        "metrics_csv": final_metrics_csv,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clean bowling test output for one or more videos.")
    parser.add_argument("--video", default=None, help="Optional path to one bowling side-on video.")
    parser.add_argument("--video_dir", default=None, help="Folder to choose a video from. Defaults to input_videos/test/bowling.")
    parser.add_argument("--output_dir", default="outputs/test_output/bowling", help="Clean bowling output folder.")
    parser.add_argument("--all", action="store_true", help="Process every bowling video in the input folder.")
    parser.add_argument("--clear", action="store_true", help="Clear old bowling test outputs before exporting.")
    parser.add_argument("--no-clear", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    output_dir = resolve_output_dir(args.output_dir)
    if args.video:
        video_paths = [Path(args.video)]
    else:
        video_dir = Path(args.video_dir) if args.video_dir else default_test_video_dir()
        ensure_input_folder(video_dir)
        print(f"[TEST OUTPUT] Input video folder: {video_dir}")
        if args.all:
            video_paths = find_videos(video_dir)
            if not video_paths:
                raise FileNotFoundError(
                    f"No videos found in: {video_dir}\n"
                    f"Put your bowling test videos in this folder, then run this file again."
                )
        else:
            video_paths = [choose_video(video_dir)]

    if args.clear and not args.no_clear:
        clear_output_folder(output_dir)
    else:
        ensure_output_folders(output_dir)

    all_results = []
    for video_path in video_paths:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        all_results.append(export_video(video_path.resolve(), output_dir, clear_output=False))

    print("\n[TEST OUTPUT] Done.")
    print(f"  Processed videos: {len(all_results)}")
    for results in all_results:
        print(f"  Video: {results['video']}")
        print(f"  Phase label CSV: {results['phase_label_csv']}")
        print(f"  Keypoints CSV: {results['keypoints_csv']}")
        print(f"  Features CSV: {results['features_csv']}")
        print(f"  Metrics CSV: {results['metrics_csv']}")


if __name__ == "__main__":
    main()
