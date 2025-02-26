import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from dataclasses import dataclass
from collections import deque
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class GaitMetrics:
    left_knee_angle: float
    right_knee_angle: float
    left_hip_angle: float
    right_hip_angle: float
    left_ankle_angle: float
    right_ankle_angle: float
    stride_length: float = 0.0
    step_width: float = 0.0
    gait_symmetry: float = 0.0
    cadence: float = 0.0
    gait_pattern: str = ""
    confidence_score: float = 0.0

class GaitProcessor:
    def __init__(self, history_length=30):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Increased complexity for better accuracy
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize metrics history with all fields from GaitMetrics
        self.metrics_history = {
            'left_knee_angle': deque(maxlen=history_length),
            'right_knee_angle': deque(maxlen=history_length),
            'left_hip_angle': deque(maxlen=history_length),
            'right_hip_angle': deque(maxlen=history_length),
            'left_ankle_angle': deque(maxlen=history_length),
            'right_ankle_angle': deque(maxlen=history_length),
            'stride_length': deque(maxlen=history_length),
            'step_width': deque(maxlen=history_length),
            'gait_symmetry': deque(maxlen=history_length),
            'cadence': deque(maxlen=history_length),
            'gait_pattern': deque(maxlen=history_length),
            'confidence_score': deque(maxlen=history_length)
        }
        
        self.frame_buffer = []
        self.step_timestamps = []
        self.known_patterns = self._load_known_patterns()
        
    def _load_known_patterns(self):
        """Load known criminal gait patterns from database."""
        try:
            df = pd.read_csv('criminal_gait_database.csv')
            patterns = {}
            for _, row in df.iterrows():
                patterns[row['suspect_id']] = {
                    'gait_signature': eval(row['gait_signature']),
                    'criminal_history': row['criminal_history']
                }
            return patterns
        except FileNotFoundError:
            return {}
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, GaitMetrics]:
        """Process a single frame and return the annotated frame and gait metrics."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        annotated_frame = frame.copy()
        metrics = None
        
        if results.pose_landmarks:
            # Draw pose with improved visibility
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Calculate metrics
            metrics = self._calculate_metrics(results.pose_landmarks, frame.shape)
            self._update_metrics_history(metrics)
            
            # Add real-time measurements to frame
            self._add_measurements_to_frame(annotated_frame, metrics)
            
        return annotated_frame, metrics
    
    def _calculate_metrics(self, landmarks, frame_shape) -> GaitMetrics:
        """Enhanced metrics calculation with criminal gait analysis."""
        # Get coordinates for all key points
        keypoints = self._get_keypoints(landmarks)
        
        # Calculate base angles
        left_knee_angle = self._calculate_angle(
            keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle'])
        right_knee_angle = self._calculate_angle(
            keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle'])
        
        left_hip_angle = self._calculate_angle(
            keypoints['left_shoulder'], keypoints['left_hip'], keypoints['left_knee'])
        right_hip_angle = self._calculate_angle(
            keypoints['right_shoulder'], keypoints['right_hip'], keypoints['right_knee'])
        
        left_ankle_angle = self._calculate_angle(
            keypoints['left_knee'], keypoints['left_ankle'], keypoints['left_foot_index'])
        right_ankle_angle = self._calculate_angle(
            keypoints['right_knee'], keypoints['right_ankle'], keypoints['right_foot_index'])
        
        # Calculate stride length and step width
        stride_length = self._calculate_stride_length(keypoints, frame_shape[1])
        step_width = self._calculate_step_width(keypoints, frame_shape[0])
        
        # Create base metrics
        metrics = GaitMetrics(
            left_knee_angle=left_knee_angle,
            right_knee_angle=right_knee_angle,
            left_hip_angle=left_hip_angle,
            right_hip_angle=right_hip_angle,
            left_ankle_angle=left_ankle_angle,
            right_ankle_angle=right_ankle_angle,
            stride_length=stride_length,
            step_width=step_width
        )
        
        # Calculate additional criminal gait analysis metrics
        metrics.gait_symmetry = self._calculate_gait_symmetry(metrics)
        metrics.cadence = self._calculate_cadence()
        metrics.gait_pattern = self._identify_gait_pattern(metrics)
        metrics.confidence_score = self._calculate_confidence_score(metrics)
        
        return metrics
    
    def _calculate_gait_symmetry(self, metrics):
        """Calculate symmetry between left and right side movements."""
        knee_symmetry = abs(metrics.left_knee_angle - metrics.right_knee_angle)
        hip_symmetry = abs(metrics.left_hip_angle - metrics.right_hip_angle)
        ankle_symmetry = abs(metrics.left_ankle_angle - metrics.right_ankle_angle)
        
        # Normalize to 0-1 range where 1 is perfect symmetry
        return 1 - (knee_symmetry + hip_symmetry + ankle_symmetry) / 300
    
    def _calculate_cadence(self):
        """Calculate steps per minute."""
        if len(self.step_timestamps) < 2:
            return 0
        
        time_diff = self.step_timestamps[-1] - self.step_timestamps[0]
        if time_diff == 0:
            return 0
            
        steps_per_second = len(self.step_timestamps) / time_diff
        return steps_per_second * 60
    
    def _identify_gait_pattern(self, metrics):
        """Identify specific gait patterns associated with criminal behavior."""
        # Create feature vector from current metrics
        current_pattern = np.array([
            metrics.stride_length,
            metrics.step_width,
            metrics.gait_symmetry,
            metrics.cadence
        ])
        
        # Compare with known patterns
        best_match = None
        highest_similarity = -1
        
        for suspect_id, pattern in self.known_patterns.items():
            known_pattern = np.array([
                pattern['gait_signature']['stride_length_avg'],
                pattern['gait_signature']['step_width_avg'],
                pattern['gait_signature']['gait_symmetry'],
                pattern['gait_signature']['cadence']
            ])
            
            similarity = cosine_similarity(
                current_pattern.reshape(1, -1),
                known_pattern.reshape(1, -1)
            )[0][0]
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = suspect_id
        
        return best_match if highest_similarity > 0.8 else None
    
    def _calculate_confidence_score(self, metrics):
        """Calculate confidence score for the gait pattern match."""
        if not metrics.gait_pattern:
            return 0.0
            
        # Weighted scoring based on multiple factors
        weights = {
            'symmetry': 0.3,
            'cadence': 0.2,
            'stride': 0.25,
            'step_width': 0.25
        }
        
        pattern = self.known_patterns[metrics.gait_pattern]['gait_signature']
        
        scores = {
            'symmetry': 1 - abs(metrics.gait_symmetry - pattern['gait_symmetry']),
            'cadence': 1 - abs(metrics.cadence - pattern['cadence']) / 100,
            'stride': 1 - abs(metrics.stride_length - pattern['stride_length_avg']) / 100,
            'step_width': 1 - abs(metrics.step_width - pattern['step_width_avg']) / 20
        }
        
        return sum(score * weights[metric] for metric, score in scores.items())
    
    def _get_keypoints(self, landmarks) -> Dict[str, np.ndarray]:
        """Extract all relevant keypoints from landmarks."""
        keypoints = {}
        landmark_dict = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE,
            'left_foot_index': self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
            'right_foot_index': self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
        }
        
        for name, landmark in landmark_dict.items():
            keypoints[name] = np.array([
                landmarks.landmark[landmark].x,
                landmarks.landmark[landmark].y,
                landmarks.landmark[landmark].z
            ])
        
        return keypoints
    
    def _calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculate the angle between three points in 3D space."""
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def _calculate_stride_length(self, keypoints: Dict[str, np.ndarray], frame_width: int) -> float:
        """Calculate stride length as the horizontal distance between feet."""
        left_foot = keypoints['left_foot_index']
        right_foot = keypoints['right_foot_index']
        return abs(left_foot[0] - right_foot[0]) * frame_width
    
    def _calculate_step_width(self, keypoints: Dict[str, np.ndarray], frame_height: int) -> float:
        """Calculate step width as the lateral distance between feet."""
        left_foot = keypoints['left_foot_index']
        right_foot = keypoints['right_foot_index']
        return abs(left_foot[1] - right_foot[1]) * frame_height
    
    def _update_metrics_history(self, metrics: GaitMetrics):
        """Update the metrics history for real-time plotting."""
        for metric_name, value in metrics.__dict__.items():
            if metric_name in self.metrics_history:
                # Convert None to 0 for numerical fields
                if value is None and metric_name not in ['gait_pattern']:
                    value = 0
            self.metrics_history[metric_name].append(value)
    
    def _add_measurements_to_frame(self, frame: np.ndarray, metrics: GaitMetrics):
        """Add real-time measurements to the frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_pos = 30
        for metric_name, value in metrics.__dict__.items():
            # Format value based on its type
            if value is None:
                formatted_value = "None"
            elif isinstance(value, float):
                formatted_value = f"{value:.1f}"
            elif isinstance(value, str):
                formatted_value = value
            else:
                formatted_value = str(value)
            
            text = f"{metric_name}: {formatted_value}"
            cv2.putText(frame, text, (10, y_pos), font, 0.6, (0, 255, 0), 2)
            y_pos += 25