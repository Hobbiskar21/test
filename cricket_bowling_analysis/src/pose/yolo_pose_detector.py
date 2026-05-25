"""
src/pose/yolo_pose_detector.py
-------------------------------
YOLO-based pose detection with person tracking.

Uses YOLO Pose (17 COCO keypoints) instead of MediaPipe.
Implements:
- Person tracking (largest bbox in frame 0, closest to locked person in subsequent frames)
- Guard against ID switches (max_jump_px threshold)
- Returns RAW YOLO detections (no smoothing)
- COCO keypoint format (17 points)

COCO Keypoint Mapping:
0=nose, 1=leye, 2=reye, 3=lear, 4=rear,
5=lsho, 6=rsho, 7=lelb, 8=relb, 9=lwri, 10=rwri,
11=lhip, 12=rhip, 13=lknee, 14=rknee, 15=lank, 16=rank
"""

import numpy as np
from ultralytics import YOLO
from typing import Tuple, Optional, List, Dict


# COCO keypoint indices
COCO_KPTS = {
    "nose": 0, "leye": 1, "reye": 2, "lear": 3, "rear": 4,
    "lsho": 5, "rsho": 6, "lelb": 7, "relb": 8, "lwri": 9,
    "rwri": 10, "lhip": 11, "rhip": 12, "lknee": 13, "rknee": 14,
    "lank": 15, "rank": 16
}


def pick_person_keypoints(result, prev_center=None, max_distance=150):
    """
    Pick the best person from YOLO detection results.
    
    Strategy:
    - First frame: pick largest bounding box
    - Subsequent frames: pick closest to previous center, BUT only if within max_distance
    
    Parameters
    ----------
    result : YOLO result
        Result from YOLO pose prediction
    prev_center : tuple, optional
        Previous person center (x, y)
    max_distance : float
        Maximum distance to consider as same person (pixels)
    
    Returns
    -------
    tuple
        (keypoints_array, center_xy) or (None, None) if no detection
    """
    if result.keypoints is None or result.keypoints.xy is None:
        return None, None

    kpts_xy = result.keypoints.xy
    boxes = getattr(result, 'boxes', None)

    if kpts_xy is None or len(kpts_xy) == 0 or boxes is None or boxes.xyxy is None:
        return None, None

    kpts_np = kpts_xy.cpu().numpy()
    bx = boxes.xyxy.cpu().numpy()
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in bx]

    cand = list(range(len(kpts_np)))

    if prev_center is not None:
        # Pick closest to previous center, but ONLY if within max_distance
        distances = [np.hypot(centers[i][0] - prev_center[0], centers[i][1] - prev_center[1]) for i in cand]
        best_idx = int(np.argmin(distances))
        best_distance = distances[best_idx]
        
        # If closest person is too far away, reject (person left frame or switched)
        if best_distance > max_distance:
            return None, None
        
        best = best_idx
    else:
        # Pick largest bounding box
        areas = [(bx[i][2] - bx[i][0]) * (bx[i][3] - bx[i][1]) for i in cand]
        best = cand[int(np.argmax(areas))]

    return kpts_np[best], centers[best]


def detect_pose_sequence(frames: List[np.ndarray], 
                        width: int, 
                        height: int,
                        pose_model: str = "yolov8s-pose.pt",
                        imgsz: int = 960,
                        max_jump_px: float = 260.0,
                        max_person_distance: float = 150.0,
                        device: Optional[str] = None) -> Tuple[List, Dict]:
    """
    Run YOLO Pose on a list of BGR frames with strict person tracking.
    
    Returns RAW YOLO detections without smoothing.
    Implements strict person tracking - locks to first person and rejects frames if person moves too far or another person gets closer.
    
    Parameters
    ----------
    frames : list
        List of BGR frames
    width, height : int
        Frame dimensions
    pose_model : str
        YOLO pose model name
    imgsz : int
        Input image size for YOLO
    max_jump_px : float
        Maximum allowed person center movement per frame (guard against ID switches)
    max_person_distance : float
        Maximum distance to consider as same person (pixels). If closest person is farther, frame is rejected.
    device : str, optional
        Device to use (cpu/cuda). If None, auto-detects GPU availability.
    
    Returns
    -------
    tuple
        (landmarks_sequence, metadata)
        - landmarks_sequence: List of (17, 2) keypoint arrays or None per frame (RAW, unsmoothed)
        - metadata: Dict with detection statistics
    """
    # Auto-detect GPU if device not specified
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    
    # Force GPU if requested
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print(f"[ERROR] GPU requested but CUDA not available")
                print(f"[ERROR] Install PyTorch with CUDA support:")
                print(f"[ERROR] pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                raise RuntimeError("GPU not available")
        except ImportError:
            print(f"[ERROR] PyTorch not installed")
            raise
    
    print(f"[POSE] Loading YOLO pose model: {pose_model}")
    print(f"[POSE] Using device: {device}")
    model = YOLO(pose_model)
    
    # Move model to device
    model.to(device)
    
    landmarks_sequence = []
    prev_center = None
    frame_with_first_detection = None
    detected_count = 0
    rejected_count = 0
    stable_track_count = 0
    
    for i, frame in enumerate(frames):
        # Run YOLO pose detection
        results = model.predict(frame, imgsz=imgsz, verbose=False)
        result = results[0]
        
        # Get best person from detections (with strict distance check)
        kpts, center = pick_person_keypoints(result, prev_center=prev_center, max_distance=max_person_distance)
        
        if kpts is None:
            print(f"[POSE] Frame {i}: NO PERSON DETECTED")
            landmarks_sequence.append(None)
            stable_track_count = 0
            continue
        
        # Guard against ID switches - check if person moved too far
        if prev_center is not None and center is not None:
            dist = np.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
            if dist > max_jump_px:
                print(f"[POSE] Frame {i}: REJECTED (person moved {dist:.0f}px, threshold {max_jump_px:.0f}px)")
                landmarks_sequence.append(None)
                rejected_count += 1
                stable_track_count = 0
                continue
        
        # Update tracking
        if prev_center is None:
            frame_with_first_detection = i
            stable_track_count = 1
            print(f"[POSE] Frame {i}: LOCKED onto person at center {center}")
        else:
            stable_track_count += 1
            dist = np.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
            print(f"[POSE] Frame {i}: TRACKING person (moved {dist:.0f}px from previous)")
        
        prev_center = center
        landmarks_sequence.append(kpts)
        detected_count += 1
        
        if i % 100 == 0:
            print(f"[POSE] Frame {i}/{len(frames)} - detected in {detected_count}/{i+1}")
    
    # NO SMOOTHING - return raw YOLO detections
    # Smoothing will be applied separately in the pipeline if needed
    
    detected = sum(1 for l in landmarks_sequence if l is not None)
    print(f"[POSE] Done. Detected in {detected}/{len(frames)} frames. Locked to person from frame {frame_with_first_detection}")
    
    metadata = {
        "detection_method": "YOLO Pose",
        "keypoint_format": "COCO (17 points)",
        "total_frames": len(frames),
        "detected_frames": detected,
        "rejected_frames": rejected_count,
        "first_detection_frame": frame_with_first_detection,
        "max_jump_px": max_jump_px,
    }
    
    return landmarks_sequence, metadata
