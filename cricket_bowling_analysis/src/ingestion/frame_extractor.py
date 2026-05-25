"""
src/ingestion/frame_extractor.py
# CAMERA: side | front | back

Extracts frames from each camera video independently.
Applies sync offsets per camera.
Truncates all cameras to the same frame count.

BACKGROUND BLURRING:
Uses OpenCV Gaussian blur to reduce noise and improve pose detection accuracy.
Blurred frames are used ONLY for analysis (pose detection, phase segmentation).
Original unblurred frames are kept for output video rendering.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


def blur_background(frame: np.ndarray, blur_strength: int = 25) -> np.ndarray:
    """
    Blur the frame using OpenCV Gaussian blur.
    Reduces noise and improves YOLO pose detection accuracy.
    
    Parameters
    ----------
    frame : np.ndarray
        BGR frame from video
    blur_strength : int
        Blur kernel size (must be odd). Higher = more blur.
        Default 25 is good for reducing noise while keeping pose features visible.
    
    Returns
    -------
    np.ndarray
        Frame with Gaussian blur applied
    """
    try:
        # Apply Gaussian blur to reduce noise
        # This helps YOLO detect cleaner keypoints
        blurred_frame = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
        return blurred_frame
    
    except Exception as e:
        # If blur fails, return original frame
        print(f"[INGESTION] Warning: Gaussian blur failed ({e}), using original frame")
        return frame


def extract_all_cameras(session_config: Dict, offsets: Dict[str, int]) -> Dict[str, Dict[str, List]]:
    """
    Extract frames from all 3 cameras with sync offsets applied.
    Returns BOTH original and blurred frames.
    
    Parameters
    ----------
    session_config : dict
        SessionConfig from session_loader with cameras dict
    offsets : dict
        Sync offsets from frame_aligner:
        {
            "side": 0,
            "front": +17,
            "back": -7,
        }
    
    Returns
    -------
    dict
        {
            "side": {
                "original": [frame0, frame1, ...],
                "blurred": [frame0_blurred, frame1_blurred, ...]
            },
            "front": {...},
            "back": {...},
        }
        All cameras truncated to same frame count.
        Skips cameras that are None in session_config.
    """
    cameras = session_config["cameras"]
    all_frames = {}
    
    print(f"\n[INGESTION] Extracting frames from all cameras...")
    
    # Extract frames from each camera
    for camera_name in ["side", "front", "back"]:
        if cameras[camera_name] is None:
            print(f"[INGESTION] Skipping {camera_name} (not available)")
            all_frames[camera_name] = None
            continue
        
        video_path = cameras[camera_name]["path"]
        offset = offsets.get(camera_name, 0)
        
        print(f"[INGESTION] Extracting {camera_name} (offset: {offset:+d} frames)...")
        
        cap = cv2.VideoCapture(video_path)
        original_frames = []
        blurred_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply offset: skip first N frames if offset > 0
            if frame_idx >= offset:
                # Keep original frame for output
                original_frames.append(frame)
                # Create blurred version for analysis
                frame_blurred = blur_background(frame, blur_strength=25)
                blurred_frames.append(frame_blurred)
            
            frame_idx += 1
        
        cap.release()
        all_frames[camera_name] = {
            "original": original_frames,
            "blurred": blurred_frames,
        }
        print(f"[INGESTION] OK {camera_name}: extracted {len(original_frames)} frames")
    
    # Truncate all cameras to same frame count
    frame_counts = [len(f["original"]) for f in all_frames.values() if f is not None]
    if frame_counts:
        min_frames = min(frame_counts)
        print(f"\n[INGESTION] Truncating all cameras to {min_frames} frames (minimum)")
        
        for camera_name in ["side", "front", "back"]:
            if all_frames[camera_name] is not None:
                all_frames[camera_name]["original"] = all_frames[camera_name]["original"][:min_frames]
                all_frames[camera_name]["blurred"] = all_frames[camera_name]["blurred"][:min_frames]
    
    return all_frames


def extract_single_camera(video_path: str, offset: int = 0) -> Tuple[List, List]:
    """
    Extract frames from a single camera video with offset.
    Returns BOTH original and blurred frames.
    
    Parameters
    ----------
    video_path : str
        Path to .mp4 file
    offset : int
        Number of frames to skip from start
    
    Returns
    -------
    tuple
        (original_frames, blurred_frames)
        Both are lists of frames
    """
    cap = cv2.VideoCapture(video_path)
    original_frames = []
    blurred_frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx >= offset:
            # Keep original frame for output
            original_frames.append(frame)
            # Create blurred version for analysis
            frame_blurred = blur_background(frame, blur_strength=25)
            blurred_frames.append(frame_blurred)
        
        frame_idx += 1
    
    cap.release()
    return original_frames, blurred_frames
