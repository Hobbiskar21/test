"""Configuration for side-on repeatability analysis.

This module is intentionally declarative. Repeatability code reads existing
frame-wise CSV features produced by the video-analysis pipeline and keeps the
pose/biomechanics/storage logic separate from repeatability calculations.
"""

PHASE_NAMES = [
    "approach",
    "jump_bound",
    "bfc_window",
    "bfc_to_ffc",
    "ffc_window",
    "ffc_to_release",
    "follow_through",
]

EVENT_NAMES = [
    "jump_start",
    "jump_peak",
    "bfc",
    "ffc",
    "release",
    "follow_through_end",
]

PHASE_ID_MAP = {
    "approach": 0,
    "jump_bound": 1,
    "bfc_window": 2,
    "bfc_to_ffc": 3,
    "ffc_window": 4,
    "ffc_to_release": 5,
    "follow_through": 6,
}

PHASE_FIXED_LENGTHS = {
    "approach": 20,
    "jump_bound": 15,
    "bfc_window": 10,
    "bfc_to_ffc": 20,
    "ffc_window": 10,
    "ffc_to_release": 25,
    "follow_through": 20,
}

PHASE_WEIGHTS = {
    "approach": 0.10,
    "jump_bound": 0.15,
    "bfc_window": 0.10,
    "bfc_to_ffc": 0.15,
    "ffc_window": 0.15,
    "ffc_to_release": 0.25,
    "follow_through": 0.10,
}

LSTM_FEATURES = [
    "front_knee_angle",
    "back_knee_angle",
    "trunk_lean_angle",
    "bowling_arm_angle",
    "bowling_elbow_angle",
    "hip_center_x",
    "hip_center_y",
    "head_x",
    "head_y",
    "wrist_speed",
    "hip_speed",
    "front_knee_angle_velocity",
    "arm_angle_velocity",
    "trunk_lean_velocity",
    "phase_id",
    "normalized_phase_time",
]

PHASE_FEATURES = {
    "approach": [
        "hip_center_x", "hip_center_y", "hip_speed", "head_x", "head_y",
        "trunk_lean_angle", "head_speed", "stride_length_proxy",
        "front_ankle_x", "back_ankle_x",
    ],
    "jump_bound": [
        "hip_center_x", "hip_center_y", "head_x", "head_y",
        "trunk_lean_angle", "front_knee_angle", "back_knee_angle",
        "front_ankle_x", "front_ankle_y", "back_ankle_x", "back_ankle_y",
        "hip_speed",
    ],
    "bfc_window": [
        "back_ankle_x", "back_ankle_y", "back_knee_angle",
        "back_leg_angle", "hip_center_y", "trunk_lean_angle",
        "head_x", "head_y", "hip_speed",
    ],
    "bfc_to_ffc": [
        "hip_center_x", "hip_center_y", "hip_speed", "front_ankle_x",
        "front_ankle_y", "back_ankle_x", "back_ankle_y",
        "front_knee_angle", "back_knee_angle", "trunk_lean_angle",
        "head_x", "head_y", "front_ankle_speed",
    ],
    "ffc_window": [
        "front_ankle_x", "front_ankle_y", "front_knee_angle",
        "front_leg_angle", "hip_center_y", "trunk_lean_angle",
        "head_x", "head_y", "hip_speed",
    ],
    "ffc_to_release": [
        "front_knee_angle", "front_knee_angle_velocity",
        "bowling_arm_angle", "arm_angle_velocity", "bowling_elbow_angle",
        "bowling_wrist_x", "bowling_wrist_y", "wrist_speed",
        "trunk_lean_angle", "trunk_lean_velocity", "head_x", "head_y",
        "hip_center_x", "hip_center_y", "release_height_proxy",
    ],
    "follow_through": [
        "head_x", "head_y", "hip_center_x", "hip_center_y",
        "trunk_lean_angle", "hip_speed", "head_speed",
        "body_center_x", "body_center_y",
    ],
}

SIDEON_FEATURES = sorted(
    set(LSTM_FEATURES)
    .union(*(set(features) for features in PHASE_FEATURES.values()))
    .difference({"phase_id", "normalized_phase_time"})
)

METADATA_COLUMNS = [
    "frame_id",
    "delivery_id",
    "bowler_id",
    "video_id",
]

DEFAULT_OUTPUT_DIR = "outputs/repeatability"

OUTPUT_SUBDIRS = {
    "input": "input",
    "auto_events": "auto_events",
    "auto_phases": "auto_phases",
    "confirmed_events": "confirmed_events",
    "confirmed_phases": "confirmed_phases",
    "sideon_features": "sideon_features",
    "phase_labelled": "phase_labelled",
    "movement_curves": "movement_curves",
    "delivery_sequences": "delivery_sequences",
    "lstm_dataset": "lstm_dataset",
    "models": "models",
    "predictions": "predictions",
    "graphs": "graphs",
    "dashboard": "dashboard",
    "scores": "scores",
}

PHASE_PRIORITY = [
    "bfc_window",
    "ffc_window",
    "ffc_to_release",
    "bfc_to_ffc",
    "jump_bound",
    "follow_through",
    "approach",
]

PHASE_DISPLAY_NAMES = {
    "approach": "Approach",
    "jump_bound": "Jump / Bound",
    "bfc_window": "BFC Window",
    "bfc_to_ffc": "BFC -> FFC",
    "ffc_window": "FFC Window",
    "ffc_to_release": "FFC -> Release",
    "follow_through": "Follow-through",
}
