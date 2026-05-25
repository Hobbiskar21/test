import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean

class BattingFeatureExtractor:
    """
    Extract batting metrics from COCO keypoints
    COCO keypoints: 0-Nose, 1-LEye, 2-REye, 3-LEar, 4-REar, 
                    5-LShoulder, 6-RShoulder, 7-LElbow, 8-RElbow,
                    9-LWrist, 10-RWrist, 11-LHip, 12-RHip,
                    13-LKnee, 14-RKnee, 15-LAnkle, 16-RAnkle
    """
    
    def __init__(self, right_handed=True):
        self.right_handed = right_handed
        # For right-handed batter: front side = left, back side = right
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
        """
        Calculate angle at p2 formed by p1-p2-p3
        Returns angle in degrees
        """
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def calculate_com_shift(self, keypoints):
        """
        Weight Transfer (COM Shift): 
        Calculate horizontal shift of center of mass between back hip and front hip
        Returns shift magnitude (0-1 normalized)
        """
        # Get hip positions
        left_hip_x, left_hip_y = self.extract_keypoint(keypoints, 'left_hip')
        right_hip_x, right_hip_y = self.extract_keypoint(keypoints, 'right_hip')
        
        # Get ankle positions for reference
        left_ankle_x, left_ankle_y = self.extract_keypoint(keypoints, 'left_ankle')
        right_ankle_x, right_ankle_y = self.extract_keypoint(keypoints, 'right_ankle')
        
        if self.right_handed:
            # For right-handed: front=left, back=right
            front_hip_x = left_hip_x
            back_hip_x = right_hip_x
            front_ankle_x = left_ankle_x
            back_ankle_x = right_ankle_x
        else:
            front_hip_x = right_hip_x
            back_hip_x = left_hip_x
            front_ankle_x = right_ankle_x
            back_ankle_x = left_ankle_x
        
        # Calculate horizontal distance between hips
        hip_distance = abs(back_hip_x - front_hip_x)
        
        # Calculate shift as ratio of front hip to total distance
        if hip_distance > 0:
            com_shift = front_hip_x / (back_hip_x + front_hip_x + 1e-6)
        else:
            com_shift = 0.5
            
        return np.clip(com_shift, 0, 1)
    
    def calculate_front_knee_angle(self, keypoints):
        """
        Front Knee Angle at Impact:
        Calculate angle at front knee joint
        Returns angle in degrees
        """
        if self.right_handed:
            # Front knee = left knee for right-hander
            knee_x, knee_y = self.extract_keypoint(keypoints, 'left_knee')
            hip_x, hip_y = self.extract_keypoint(keypoints, 'left_hip')
            ankle_x, ankle_y = self.extract_keypoint(keypoints, 'left_ankle')
        else:
            knee_x, knee_y = self.extract_keypoint(keypoints, 'right_knee')
            hip_x, hip_y = self.extract_keypoint(keypoints, 'right_hip')
            ankle_x, ankle_y = self.extract_keypoint(keypoints, 'right_ankle')
        
        hip_point = (hip_x, hip_y)
        knee_point = (knee_x, knee_y)
        ankle_point = (ankle_x, ankle_y)
        
        angle = self.calculate_angle(hip_point, knee_point, ankle_point)
        return angle
    
    def extract_features_single_frame(self, keypoints):
        """
        Extract all features from single frame
        keypoints: 34-length array (17 joints * 2 for x,y)
        """
        features = {}
        
        try:
            features['weight_transfer'] = self.calculate_com_shift(keypoints)
            features['front_knee_angle'] = self.calculate_front_knee_angle(keypoints)
            features['head_stability_raw'] = 1.0  # Will be filled in batch processing
        except Exception as e:
            features['weight_transfer'] = np.nan
            features['front_knee_angle'] = np.nan
            features['head_stability_raw'] = np.nan
        
        return features
    
    def calculate_head_stability_index(self, keypoints_sequence):
        """
        Head Stability Index:
        Calculate variance/stability of head position across a sequence of frames
        Lower variance = more stable
        Returns stability index (0-1, where 1 is most stable)
        keypoints_sequence: list of keypoint arrays
        """
        head_positions = []
        
        for keypoints in keypoints_sequence:
            try:
                nose_x, nose_y = self.extract_keypoint(keypoints, 'nose')
                head_positions.append((nose_x, nose_y))
            except:
                continue
        
        if len(head_positions) < 2:
            return 0.5
        
        head_positions = np.array(head_positions)
        
        # Calculate variance in x and y
        var_x = np.var(head_positions[:, 0])
        var_y = np.var(head_positions[:, 1])
        total_var = var_x + var_y
        
        # Normalize to 0-1 (inverse: lower variance = higher stability)
        # Using max possible variance as reference
        max_var = (np.max(head_positions[:, 0]) - np.min(head_positions[:, 0]))**2 + \
                  (np.max(head_positions[:, 1]) - np.min(head_positions[:, 1]))**2
        
        if max_var > 0:
            stability_index = 1 - (total_var / max_var)
        else:
            stability_index = 1.0
        
        return np.clip(stability_index, 0, 1)
    
    def extract_features_batch(self, keypoints_list):
        """
        Extract features from batch of frames
        keypoints_list: list of keypoint arrays
        Returns: list of feature dicts
        """
        features_list = []
        
        for keypoints in keypoints_list:
            features = self.extract_features_single_frame(keypoints)
            features_list.append(features)
        
        # Calculate head stability for the entire sequence
        head_stability = self.calculate_head_stability_index(keypoints_list)
        for feature_dict in features_list:
            feature_dict['head_stability_index'] = head_stability
        
        return features_list
