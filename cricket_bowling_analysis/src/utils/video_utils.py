"""
src/utils/video_utils.py
--------------------------
Shared OpenCV helpers used across the project.
- get_video_info() : read fps, width, height, frame count
- extract_frames() : read frames into a list
- write_video()    : write annotated frames to .mp4
- frame_to_rgb()   : BGR to RGB conversion
"""

import os
from typing import Optional

import cv2
import numpy as np


def get_video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    info = {
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration_sec"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def extract_frames(path: str,
                   start_frame: int = 0,
                   end_frame: Optional[int] = None) -> list:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx < start_frame:
            idx += 1
            continue
        if end_frame is not None and idx >= end_frame:
            break
        frames.append(frame)
        idx += 1
    cap.release()
    return frames


def write_video(frames: list, output_path: str, fps: float, slowdown: float = 1.0) -> bool:
    """
    Write frames to video file.
    
    Args:
        frames: List of frame arrays
        output_path: Path to save video
        fps: Original video FPS
        slowdown: Slowdown factor (0.25 = 4x slower, 0.5 = 2x slower, 1.0 = normal)
    
    Returns:
        bool: True if video was written successfully, False otherwise
    """
    if not frames:
        print(f"[ERROR] No frames to write to {output_path}")
        return False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[VIDEO] Created output directory: {output_dir}")
        except Exception as e:
            print(f"[ERROR] Failed to create output directory {output_dir}: {e}")
            return False
    
    first_frame = frames[0]
    if first_frame is None or len(first_frame.shape) < 2:
        print(f"[ERROR] First frame is invalid; cannot write video")
        return False

    h, w = first_frame.shape[:2]
    # Adjust FPS based on slowdown factor
    output_fps = max(float(fps or 25.0) * float(slowdown or 1.0), 1.0)
    
    # Determine output format and codec
    if output_path.lower().endswith('.avi'):
        codec_candidates = ["MJPG", "XVID"]
        print(f"[VIDEO] Using AVI output")
    else:
        output_path = output_path if output_path.lower().endswith('.mp4') else output_path + '.mp4'
        codec_candidates = ["mp4v", "avc1", "H264", "MJPG"]
        print(f"[VIDEO] Using MP4 output")
    
    print(f"[VIDEO] Writing {len(frames)} frames to {output_path}")
    print(f"[VIDEO] Resolution: {w}x{h}, FPS: {output_fps:.1f}")
    
    writer = None
    active_codec = None
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))
        if writer.isOpened():
            active_codec = codec
            print(f"[VIDEO] Using {codec} codec")
            break
        writer.release()

    if writer is None or not writer.isOpened():
        print(f"[ERROR] Failed to open VideoWriter with codecs: {', '.join(codec_candidates)}")
        print(f"[ERROR] Check codec support and file permissions")
        return False
    
    frame_count = 0
    for i, frame in enumerate(frames):
        if frame is None:
            print(f"[WARNING] Skipping empty frame {i}")
            continue

        safe_frame = np.asarray(frame)
        if safe_frame.dtype != np.uint8:
            safe_frame = np.clip(safe_frame, 0, 255).astype(np.uint8)

        if safe_frame.shape[:2] != (h, w):
            safe_frame = cv2.resize(safe_frame, (w, h), interpolation=cv2.INTER_LINEAR)

        if len(safe_frame.shape) == 2:
            safe_frame = cv2.cvtColor(safe_frame, cv2.COLOR_GRAY2BGR)
        elif safe_frame.shape[2] == 4:
            safe_frame = cv2.cvtColor(safe_frame, cv2.COLOR_BGRA2BGR)

        writer.write(safe_frame)
        frame_count += 1
        if (i + 1) % 100 == 0:
            print(f"[VIDEO] Wrote {i + 1}/{len(frames)} frames")
    
    writer.release()
    
    # Verify file was created
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size <= 0 or frame_count == 0:
            print(f"[ERROR] Video file was created but contains no usable frames: {output_path}")
            return False
        print(f"[VIDEO] Video saved successfully: {output_path}")
        print(f"[VIDEO] File size: {file_size / 1e6:.1f} MB")
        print(f"[VIDEO] Frames written: {frame_count} using {active_codec}")
        return True
    else:
        print(f"[ERROR] Video file was not created: {output_path}")
        return False


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def rgb_to_frame(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
