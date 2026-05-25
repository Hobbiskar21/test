import numpy as np
from scipy.signal import find_peaks

class BattingPhaseDetector:
    """
    Detect batting phases from COCO keypoint sequence
    Phases: Setup -> Loading -> Downswing -> Impact -> Follow-through
    """
    
    PHASES = {
        0: 'Setup',
        1: 'Loading',
        2: 'Downswing',
        3: 'Impact',
        4: 'Follow-through'
    }
    
    def __init__(self, right_handed=True):
        self.right_handed = right_handed
        self.front_side = 'left' if right_handed else 'right'
        self.back_side = 'right' if right_handed else 'left'
    
    def extract_keypoint(self, keypoints, joint_name):
        """Extract x, y coordinates of a joint"""
        joint_map = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
        }
        idx = joint_map[joint_name]
        return keypoints[idx*2], keypoints[idx*2 + 1]
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2"""
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_bat_angle(self, keypoints):
        """
        Estimate bat angle using wrist and elbow positions
        Higher angle = bat raised upward (loading phase)
        """
        if self.right_handed:
            elbow_x, elbow_y = self.extract_keypoint(keypoints, 'right_elbow')
            wrist_x, wrist_y = self.extract_keypoint(keypoints, 'right_wrist')
        else:
            elbow_x, elbow_y = self.extract_keypoint(keypoints, 'left_elbow')
            wrist_x, wrist_y = self.extract_keypoint(keypoints, 'left_wrist')
        
        # Angle of arm from horizontal
        dx = wrist_x - elbow_x
        dy = wrist_y - elbow_y
        
        if dx == 0:
            angle = 90 if dy > 0 else -90
        else:
            angle = np.degrees(np.arctan(dy / dx))
        
        return angle
    
    def calculate_hip_rotation(self, keypoints, baseline_keypoints=None):
        """
        Calculate hip rotation angle
        Baseline used to measure rotation from setup position
        """
        left_hip_x, left_hip_y = self.extract_keypoint(keypoints, 'left_hip')
        right_hip_x, right_hip_y = self.extract_keypoint(keypoints, 'right_hip')
        
        left_shoulder_x, left_shoulder_y = self.extract_keypoint(keypoints, 'left_shoulder')
        right_shoulder_x, right_shoulder_y = self.extract_keypoint(keypoints, 'right_shoulder')
        
        # Vector from right hip to left hip
        hip_vector = np.array([left_hip_x - right_hip_x, left_hip_y - right_hip_y])
        shoulder_vector = np.array([left_shoulder_x - right_shoulder_x, left_shoulder_y - right_shoulder_y])
        
        # Calculate angle between hip and shoulder (rotation indicator)
        if np.linalg.norm(hip_vector) > 0 and np.linalg.norm(shoulder_vector) > 0:
            cos_angle = np.dot(hip_vector, shoulder_vector) / \
                       (np.linalg.norm(hip_vector) * np.linalg.norm(shoulder_vector) + 1e-6)
            rotation = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        else:
            rotation = 0
        
        return rotation
    
    def calculate_front_knee_position(self, keypoints):
        """
        Get front knee vertical position
        Used to detect knee drive during downswing
        """
        if self.right_handed:
            knee_x, knee_y = self.extract_keypoint(keypoints, 'left_knee')
        else:
            knee_x, knee_y = self.extract_keypoint(keypoints, 'right_knee')
        
        return knee_y
    
    def extract_temporal_features(self, keypoints_sequence):
        """
        Extract features for phase detection from sequence
        """
        bat_angles = []
        hip_rotations = []
        front_knee_positions = []
        
        for keypoints in keypoints_sequence:
            bat_angles.append(self.calculate_bat_angle(keypoints))
            hip_rotations.append(self.calculate_hip_rotation(keypoints))
            front_knee_positions.append(self.calculate_front_knee_position(keypoints))
        
        return np.array(bat_angles), np.array(hip_rotations), np.array(front_knee_positions)
    
    def detect_phases(self, keypoints_sequence):
        """
        Detect batting phases from keypoint sequence
        Returns: list of phase labels for each frame
        """
        n_frames = len(keypoints_sequence)
        phases = np.zeros(n_frames, dtype=int)
        
        # Extract temporal features
        bat_angles, hip_rotations, front_knee_positions = self.extract_temporal_features(keypoints_sequence)
        
        # Smooth signals
        bat_angles_smooth = self._smooth_signal(bat_angles, window=5)
        hip_rotations_smooth = self._smooth_signal(hip_rotations, window=5)
        knee_positions_smooth = self._smooth_signal(front_knee_positions, window=5)
        
        # Find key events
        # Peak bat angle indicates loading phase
        bat_peaks, _ = find_peaks(bat_angles_smooth, prominence=5)
        
        # Valley in hip rotation indicates impact
        hip_valleys, _ = find_peaks(-hip_rotations_smooth, prominence=5)
        
        # Peak in front knee height (lowest y) indicates weight transfer completion
        knee_peaks, _ = find_peaks(-knee_positions_smooth, prominence=5)
        
        # Assign phases based on temporal position
        if len(bat_peaks) > 0:
            loading_idx = bat_peaks[0] if len(bat_peaks) > 0 else n_frames // 3
        else:
            loading_idx = n_frames // 3
        
        if len(hip_valleys) > 0:
            impact_idx = hip_valleys[0] if len(hip_valleys) > 0 else 2 * n_frames // 3
        else:
            impact_idx = 2 * n_frames // 3
        
        # Define phase boundaries
        downswing_start = int(loading_idx + (impact_idx - loading_idx) * 0.3)
        downswing_end = int(impact_idx)
        followthrough_start = downswing_end + 1
        
        # Assign phases
        for i in range(n_frames):
            if i < loading_idx:
                phases[i] = 0  # Setup
            elif i < downswing_start:
                phases[i] = 1  # Loading
            elif i < downswing_end:
                phases[i] = 2  # Downswing
            elif i < followthrough_start + n_frames // 8:
                phases[i] = 3  # Impact
            else:
                phases[i] = 4  # Follow-through
        
        return phases
    
    def _smooth_signal(self, signal, window=5):
        """Simple moving average smoothing"""
        if len(signal) < window:
            return signal
        
        smoothed = np.convolve(signal, np.ones(window)/window, mode='same')
        return smoothed
    
    def get_phase_name(self, phase_code):
        """Get phase name from code"""
        return self.PHASES.get(phase_code, 'Unknown')
    
    def detect_impact_frame(self, keypoints_sequence):
        """
        Detect the frame closest to ball impact
        Uses hip alignment and bat angle convergence
        """
        phases = self.detect_phases(keypoints_sequence)
        impact_frames = np.where(phases == 3)[0]
        
        if len(impact_frames) > 0:
            # Return middle frame of impact phase
            return impact_frames[len(impact_frames) // 2]
        else:
            return len(keypoints_sequence) // 2
