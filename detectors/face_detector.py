"""
MediaPipe Face Detection and Head Pose Estimation Module
Detects face, tracks head pose, and estimates eye gaze
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict
import logging
import math

logger = logging.getLogger(__name__)


class FaceDetector:
    """MediaPipe-based face detection and head pose estimation"""
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize MediaPipe face detector
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Face mesh indices for key points
        # Nose tip, chin, left eye, right eye, left mouth, right mouth
        self.face_landmarks_indices = {
            'nose_tip': 1,
            'chin': 175,
            'left_eye': 33,
            'right_eye': 263,
            'left_mouth': 61,
            'right_mouth': 291,
            'forehead': 10
        }
        
        logger.info("MediaPipe Face Mesh initialized")
    
    def detect_face(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict]]:
        """
        Detect face in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (face_detected, face_data)
            face_data: Dict with landmarks, bbox, etc.
        """
        if frame is None or frame.size == 0:
            logger.warning("Empty or None frame passed to detect_face")
            return False, None
        
        logger.debug(f"Face detection: frame shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            logger.info(f"❌ No face landmarks detected by MediaPipe")
            logger.info(f"   Frame shape: {frame.shape}")
            logger.info(f"   Frame dtype: {frame.dtype}")
            logger.info(f"   Frame min/max: {frame.min()}/{frame.max()}")
            logger.info(f"   Results object: {type(results)}")
            logger.info(f"   Has multi_face_landmarks: {hasattr(results, 'multi_face_landmarks')}")
            return False, None
        
        # Get first face (assuming single face)
        face_landmarks = results.multi_face_landmarks[0]
        logger.info(f"✅ MediaPipe detected face landmarks: {len(face_landmarks.landmark)}")
        
        # Additional validation: Check if detected face is reasonable
        # MediaPipe can sometimes detect false positives (like hands, objects covering camera)
        h, w = frame.shape[:2]
        logger.info(f"   Frame dimensions: {w}x{h}")
        
        # Check if we have enough landmarks (should be 468 for full face mesh)
        # Reduced threshold from 100 to 50 to be less strict
        landmark_count = len(face_landmarks.landmark)
        if landmark_count < 50:
            logger.info(f"❌ Too few landmarks detected: {landmark_count} (min: 50)")
            return False, None
        logger.info(f"   Landmark count OK: {landmark_count}")
        
        # Calculate bounding box to validate size
        x_coords = [lm.x for lm in face_landmarks.landmark]
        y_coords = [lm.y for lm in face_landmarks.landmark]
        
        face_width = (max(x_coords) - min(x_coords)) * w
        face_height = (max(y_coords) - min(y_coords)) * h
        
        # Face should be at least 30x30 pixels to be considered valid (reduced from 50x50)
        logger.info(f"   Face dimensions: {face_width:.1f}x{face_height:.1f} pixels")
        if face_width < 30 or face_height < 30:
            logger.info(f"❌ Face too small: {face_width:.1f}x{face_height:.1f} pixels (min: 30x30)")
            return False, None
        logger.info(f"   Face size OK: {face_width:.1f}x{face_height:.1f}")
        
        # Additional check: If face covers too much of the frame, might be hand/object covering camera
        face_area_ratio = (face_width * face_height) / (w * h)
        logger.info(f"   Face area ratio: {face_area_ratio:.2%}")
        if face_area_ratio > 0.85:  # Face shouldn't cover more than 85% of frame (increased from 80%)
            logger.info(f"❌ Face area too large ({face_area_ratio:.2%}), might be hand/object covering camera")
            return False, None
        logger.info(f"   Face area ratio OK: {face_area_ratio:.2%}")
        
        # Extract key points
        landmarks = {}
        
        for name, idx in self.face_landmarks_indices.items():
            landmark = face_landmarks.landmark[idx]
            landmarks[name] = {
                'x': int(landmark.x * w),
                'y': int(landmark.y * h),
                'z': landmark.z
            }
        
        # Calculate bounding box
        bbox = {
            'x_min': int(min(x_coords) * w),
            'y_min': int(min(y_coords) * h),
            'x_max': int(max(x_coords) * w),
            'y_max': int(max(y_coords) * h)
        }
        
        face_data = {
            'landmarks': landmarks,
            'bbox': bbox,
            'all_landmarks': face_landmarks
        }
        
        logger.info(f"✅ VALID FACE DETECTED")
        logger.info(f"   Dimensions: {face_width:.1f}x{face_height:.1f} pixels")
        logger.info(f"   Area ratio: {face_area_ratio:.2%}")
        logger.info(f"   Bbox: {face_data['bbox']}")
        return True, face_data
    
    def estimate_head_pose(self, face_data: Dict) -> Dict[str, float]:
        """
        Estimate head pose (pitch, yaw, roll)
        
        Args:
            face_data: Face data from detect_face
            
        Returns:
            Dict with 'pitch', 'yaw', 'roll' in degrees
        """
        if not face_data:
            return {'pitch': 0, 'yaw': 0, 'roll': 0}
        
        landmarks = face_data['landmarks']
        
        # Get 3D points (using image coordinates and estimated depth)
        image_points = np.array([
            [landmarks['nose_tip']['x'], landmarks['nose_tip']['y']],
            [landmarks['chin']['x'], landmarks['chin']['y']],
            [landmarks['left_eye']['x'], landmarks['left_eye']['y']],
            [landmarks['right_eye']['x'], landmarks['right_eye']['y']],
            [landmarks['left_mouth']['x'], landmarks['left_mouth']['y']],
            [landmarks['right_mouth']['x'], landmarks['right_mouth']['y']]
        ], dtype=np.float32)
        
        # 3D model points (approximate face model)
        model_points = np.array([
            [0.0, 0.0, 0.0],           # Nose tip
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye
            [225.0, 170.0, -135.0],    # Right eye
            [-150.0, -150.0, -125.0],  # Left mouth
            [150.0, -150.0, -125.0]    # Right mouth
        ], dtype=np.float32)
        
        # Camera parameters (approximate)
        focal_length = 500.0
        center = (image_points[:, 0].mean(), image_points[:, 1].mean())
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs
        )
        
        if not success:
            return {'pitch': 0, 'yaw': 0, 'roll': 0}
        
        # Convert rotation vector to Euler angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Extract angles
        pitch = math.degrees(math.asin(-rotation_matrix[2][1]))
        yaw = math.degrees(math.atan2(rotation_matrix[2][0], rotation_matrix[2][2]))
        roll = math.degrees(math.atan2(rotation_matrix[0][1], rotation_matrix[1][1]))
        
        return {
            'pitch': pitch,  # Nodding up/down
            'yaw': yaw,      # Turning left/right
            'roll': roll     # Tilting left/right
        }
    
    def classify_head_pose(
        self,
        head_pose: Dict[str, float],
        down_threshold: float = 25.0,
        left_threshold: float = 30.0,
        right_threshold: float = 30.0,
        away_threshold: float = 45.0
    ) -> Dict[str, bool]:
        """
        Classify head pose into violations
        
        Args:
            head_pose: Head pose angles
            down_threshold: Threshold for looking down (pitch)
            left_threshold: Threshold for looking left (yaw)
            right_threshold: Threshold for looking right (yaw)
            away_threshold: Threshold for looking away (yaw)
            
        Returns:
            Dict with violation flags
        """
        pitch = head_pose['pitch']
        yaw = head_pose['yaw']
        
        return {
            'looking_down': pitch < -down_threshold,
            'looking_left': yaw < -left_threshold,
            'looking_right': yaw > right_threshold,
            'looking_away': abs(yaw) > away_threshold
        }
    
    def estimate_eye_gaze(self, face_data: Dict) -> Dict[str, float]:
        """
        Estimate eye gaze direction (simplified)
        
        Args:
            face_data: Face data from detect_face
            
        Returns:
            Dict with gaze estimates
        """
        if not face_data:
            return {'gaze_x': 0, 'gaze_y': 0, 'attention_score': 0}
        
        # Simplified gaze estimation based on head pose
        # In production, use more sophisticated methods
        head_pose = self.estimate_head_pose(face_data)
        
        # Normalize to attention score (0-1)
        attention_score = 1.0 - min(
            abs(head_pose['yaw']) / 45.0,
            abs(head_pose['pitch']) / 30.0,
            1.0
        )
        
        return {
            'gaze_x': head_pose['yaw'],
            'gaze_y': head_pose['pitch'],
            'attention_score': max(0, attention_score)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
