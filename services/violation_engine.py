"""
Violation Engine
Orchestrates all detection modules and tracks violations
"""

import cv2
import numpy as np
import base64
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from config.settings import get_settings
from detectors.person_detector import PersonDetector
from detectors.face_detector import FaceDetector
from detectors.blur_detector import BlurDetector

logger = logging.getLogger(__name__)


class ViolationEngine:
    """Main violation detection and tracking engine"""
    
    def __init__(self):
        """Initialize violation engine with all detectors"""
        self.settings = get_settings()
        
        # Initialize detectors
        self.person_detector = PersonDetector(
            model_path=self.settings.YOLO_MODEL_PATH,
            confidence=self.settings.YOLO_CONFIDENCE
        )
        self.face_detector = FaceDetector(
            min_detection_confidence=self.settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=self.settings.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        self.blur_detector = BlurDetector(threshold=self.settings.BLUR_THRESHOLD)
        
        # Session state tracking
        self.sessions: Dict[str, Dict] = defaultdict(lambda: {
            'violations': [],
            'violation_count': 0,
            'state': {
                'multiple_person_start': None,
                'face_missing_start': None,
                'head_pose_violation_start': None,
                'blur_start': None,
                'last_head_pose': None,
                'frame_count': 0
            },
            'events': [],
            'auto_flag_triggered': False,
            'auto_submit_triggered': False,
            'last_face_detected': False,  # Track last known face detection state
            'last_face_missing_violation_time': None,  # Track when last face_missing violation was created
            'last_multiple_person_violation_time': None  # Track when last multiple_persons violation was created
        })
    
    def _decode_frame(self, frame_data: str) -> Optional[np.ndarray]:
        """
        Decode base64 frame data to OpenCV image
        
        Args:
            frame_data: Base64 encoded image string
            
        Returns:
            Decoded frame or None
        """
        try:
            logger.debug(f"Decoding frame - input length: {len(frame_data) if frame_data else 0}")
            
            # Remove data URL prefix if present
            original_length = len(frame_data) if frame_data else 0
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
                logger.debug(f"Removed data URL prefix, new length: {len(frame_data)}")
            
            # Decode base64
            logger.debug(f"Decoding base64, string length: {len(frame_data)}")
            image_bytes = base64.b64decode(frame_data)
            logger.debug(f"Base64 decoded, bytes length: {len(image_bytes)}")
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            logger.debug(f"NumPy array created, length: {len(nparr)}")
            
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("❌ Failed to decode frame - cv2.imdecode returned None")
                logger.error(f"   Input length: {original_length}")
                logger.error(f"   Base64 bytes length: {len(image_bytes)}")
                logger.error(f"   NumPy array length: {len(nparr)}")
                return None
            
            if frame.size == 0:
                logger.error("❌ Decoded frame is empty")
                logger.error(f"   Frame shape: {frame.shape}")
                return None
            
            logger.info(f"✅ Frame decoded successfully: shape={frame.shape}, dtype={frame.dtype}, size={frame.size}, min={frame.min()}, max={frame.max()}")
            
            # Debug: Save first few frames to disk for inspection (only in debug mode)
            if self.settings.DEBUG and hasattr(self, '_debug_frame_count'):
                self._debug_frame_count = getattr(self, '_debug_frame_count', 0)
                if self._debug_frame_count < 3:
                    import os
                    debug_dir = "debug_frames"
                    os.makedirs(debug_dir, exist_ok=True)
                    debug_path = os.path.join(debug_dir, f"frame_{self._debug_frame_count}.jpg")
                    cv2.imwrite(debug_path, frame)
                    logger.info(f"Saved debug frame to {debug_path}")
                    self._debug_frame_count += 1
            
            return frame
        except Exception as e:
            logger.error(f"Error decoding frame: {e}", exc_info=True)
            return None
    
    async def process_frame(
        self,
        frame_data: str,
        session_id: str,
        metadata: Dict = None
    ) -> Dict:
        """
        Process a single frame through all detectors
        
        Args:
            frame_data: Base64 encoded frame
            session_id: Session identifier
            metadata: Additional metadata (timestamp, etc.)
            
        Returns:
            Detection results and violations
        """
        session = self.sessions[session_id]
        session['state']['frame_count'] += 1
        
        # Skip frames if processing interval > 1
        if session['state']['frame_count'] % self.settings.FRAME_PROCESSING_INTERVAL != 0:
            # For skipped frames, use last known detection state or defaults
            # This ensures we always have valid detection values
            last_face_detected = session.get('last_face_detected', False)
            return {
                'status': 'skipped',
                'session_id': session_id,
                'frame_count': session['state']['frame_count'],
                'detections': {
                    'face_detected': bool(last_face_detected),  # Ensure boolean type
                    'person_count': 0,  # Default to 0 when skipped
                    'is_blurred': False,  # Default to False when skipped
                    'head_pose': None,
                    'head_pose_violations': {}
                },
                'violations': [],
                'violation_count': int(session['violation_count']),
                'auto_actions': {'auto_flag': False, 'auto_submit': False},
                'timestamp': datetime.now().isoformat()
            }
        
        # Decode frame
        logger.info("🔍 STEP 1: DECODING FRAME")
        logger.info(f"   Session: {session_id}")
        logger.info(f"   Frame data length: {len(frame_data) if frame_data else 0}")
        logger.info(f"   Frame data type: {type(frame_data)}")
        logger.info(f"   Frame data preview: {frame_data[:100] if frame_data else 'EMPTY'}...")
        
        frame = self._decode_frame(frame_data)
        if frame is None:
            logger.error(f"❌ FAILED TO DECODE FRAME for session {session_id}")
            logger.error(f"   Frame data was: {frame_data[:200] if frame_data else 'EMPTY'}")
            return {
                'status': 'error',
                'message': 'Failed to decode frame',
                'session_id': session_id,
                'detections': {
                    'face_detected': bool(session.get('last_face_detected', False)),  # Use last known state
                    'person_count': 0,
                    'is_blurred': False,
                    'head_pose': None,
                    'head_pose_violations': {}
                },
                'violations': [],
                'violation_count': int(session['violation_count']),
                'auto_actions': {'auto_flag': False, 'auto_submit': False},
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Frame decoded successfully for session {session_id}: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
        
        current_time = datetime.now()
        violations = []
        detection_results = {
            'face_detected': False,
            'person_count': 0,
            'is_blurred': False,
            'head_pose': None,
            'head_pose_violations': {}
        }
        
        # 1. Multiple Person Detection
        logger.info(f"Running person detection for session {session_id}...")
        person_count, person_detections = self.person_detector.detect(frame)
        detection_results['person_count'] = person_count
        
        logger.info(f"Person detection result: person_count={person_count}, detections={person_detections} (session: {session_id})")
        
        if person_count > 1:
            logger.warning(f"MULTIPLE PERSONS detected: {person_count} people in frame (session: {session_id})")
            if session['state']['multiple_person_start'] is None:
                session['state']['multiple_person_start'] = current_time
                logger.info(f"Multiple person violation timer started for session {session_id}")
            else:
                elapsed = (current_time - session['state']['multiple_person_start']).total_seconds()
                logger.info(f"Multiple person duration: {elapsed:.1f}s (threshold: {self.settings.MULTIPLE_PERSON_VIOLATION_SECONDS}s) for session {session_id}")
                if elapsed >= self.settings.MULTIPLE_PERSON_VIOLATION_SECONDS:
                    # Check if we already created a violation recently to avoid duplicates
                    last_violation_time = session.get('last_multiple_person_violation_time')
                    if last_violation_time is None or (current_time - last_violation_time).total_seconds() >= self.settings.MULTIPLE_PERSON_VIOLATION_SECONDS:
                        violation = {
                            'type': 'multiple_persons',
                            'timestamp': current_time.isoformat(),
                            'severity': 'high',
                            'details': {
                                'person_count': person_count,
                                'duration': elapsed
                            }
                        }
                        violations.append(violation)
                        self._add_violation(session_id, violation)
                        session['last_multiple_person_violation_time'] = current_time
                        logger.warning(f"MULTIPLE_PERSONS violation created for session {session_id} after {elapsed:.1f}s - SENDING TO FRONTEND")
        else:
            session['state']['multiple_person_start'] = None
        
        # 2. Face Detection and Head Pose
        logger.info("🔍 STEP 3: RUNNING FACE DETECTION (MediaPipe)")
        logger.info(f"   Session: {session_id}")
        logger.info(f"   Frame shape: {frame.shape}")
        
        face_detected, face_data = self.face_detector.detect_face(frame)
        detection_results['face_detected'] = face_detected
        # Store last known face detection state for skipped frames
        session['last_face_detected'] = face_detected
        
        logger.info(f"✅ FACE DETECTION RESULT")
        logger.info(f"   face_detected: {face_detected} (type: {type(face_detected)})")
        logger.info(f"   face_data exists: {face_data is not None}")
        if face_data:
            logger.info(f"   face_data keys: {face_data.keys() if isinstance(face_data, dict) else 'N/A'}")
            if isinstance(face_data, dict) and 'bbox' in face_data:
                logger.info(f"   face bbox: {face_data['bbox']}")
        
        if not face_detected:
            # Face missing
            if session['state']['face_missing_start'] is None:
                session['state']['face_missing_start'] = current_time
                logger.info(f"Face missing detected - starting timer for session {session_id}")
            else:
                elapsed = (current_time - session['state']['face_missing_start']).total_seconds()
                logger.info(f"Face missing duration: {elapsed:.1f}s (threshold: {self.settings.FACE_MISSING_VIOLATION_SECONDS}s) for session {session_id}")
                if elapsed >= self.settings.FACE_MISSING_VIOLATION_SECONDS:
                    # Check if we already created a violation for this duration to avoid duplicates
                    # Only create violation if enough time has passed since last violation
                    last_violation_time = session.get('last_face_missing_violation_time')
                    if last_violation_time is None or (current_time - last_violation_time).total_seconds() >= self.settings.FACE_MISSING_VIOLATION_SECONDS:
                        violation = {
                            'type': 'face_missing',
                            'timestamp': current_time.isoformat(),
                            'severity': 'high',
                            'details': {'duration': elapsed}
                        }
                        violations.append(violation)
                        self._add_violation(session_id, violation)
                        session['last_face_missing_violation_time'] = current_time
                        logger.warning(f"FACE_MISSING violation created for session {session_id} after {elapsed:.1f}s")
        else:
            if session['state']['face_missing_start'] is not None:
                logger.debug(f"Face detected again - resetting face_missing timer for session {session_id}")
            session['state']['face_missing_start'] = None
            
            # Head pose estimation
            head_pose = self.face_detector.estimate_head_pose(face_data)
            detection_results['head_pose'] = head_pose
            
            # Classify head pose violations
            pose_violations = self.face_detector.classify_head_pose(
                head_pose,
                down_threshold=self.settings.HEAD_DOWN_THRESHOLD,
                left_threshold=self.settings.HEAD_LEFT_THRESHOLD,
                right_threshold=self.settings.HEAD_RIGHT_THRESHOLD,
                away_threshold=self.settings.HEAD_AWAY_THRESHOLD
            )
            detection_results['head_pose_violations'] = pose_violations
            
            # Check for sustained head pose violations
            has_violation = any(pose_violations.values())
            
            if has_violation:
                if session['state']['head_pose_violation_start'] is None:
                    session['state']['head_pose_violation_start'] = current_time
                    session['state']['last_head_pose'] = pose_violations
                else:
                    # Check if same violation type persists
                    if session['state']['last_head_pose'] == pose_violations:
                        elapsed = (current_time - session['state']['head_pose_violation_start']).total_seconds()
                        if elapsed >= self.settings.HEAD_POSE_VIOLATION_SECONDS:
                            violation_type = next(
                                k for k, v in pose_violations.items() if v
                            )
                            violation = {
                                'type': f'head_pose_{violation_type}',
                                'timestamp': current_time.isoformat(),
                                'severity': 'medium',
                                'details': {
                                    'pose': head_pose,
                                    'violation_type': violation_type,
                                    'duration': elapsed
                                }
                            }
                            violations.append(violation)
                            self._add_violation(session_id, violation)
            else:
                session['state']['head_pose_violation_start'] = None
                session['state']['last_head_pose'] = None
        
        # 3. Blur Detection
        is_blurred, blur_variance = self.blur_detector.detect_blur(frame)
        detection_results['is_blurred'] = is_blurred
        detection_results['blur_variance'] = blur_variance
        
        logger.info(f"Blur detection: is_blurred={is_blurred}, variance={blur_variance:.2f}")
        
        if is_blurred:
            if session['state']['blur_start'] is None:
                session['state']['blur_start'] = current_time
                logger.info(f"Blur detected - starting timer for session {session_id}")
            else:
                elapsed = (current_time - session['state']['blur_start']).total_seconds()
                logger.info(f"Blur duration: {elapsed:.1f}s (threshold: {self.settings.BLUR_VIOLATION_SECONDS}s)")
                if elapsed >= self.settings.BLUR_VIOLATION_SECONDS:
                    violation = {
                        'type': 'blurred_camera',
                        'timestamp': current_time.isoformat(),
                        'severity': 'medium',
                        'details': {
                            'variance': blur_variance,
                            'duration': elapsed
                        }
                    }
                    violations.append(violation)
                    self._add_violation(session_id, violation)
                    logger.warning(f"BLURRED_CAMERA violation created for session {session_id} after {elapsed:.1f}s")
        else:
            if session['state']['blur_start'] is not None:
                logger.debug(f"Blur cleared - resetting timer for session {session_id}")
            session['state']['blur_start'] = None
        
        # Check for auto-flag/auto-submit
        auto_actions = self._check_auto_actions(session_id)
        
        # Debug logging
        logger.info(
            f"Frame processed for session {session_id}: "
            f"violations={len(violations)}, "
            f"violation_count={session['violation_count']}, "
            f"face_detected={detection_results['face_detected']}, "
            f"person_count={detection_results['person_count']}"
        )
        
        # Ensure all detection fields are present and properly typed
        response = {
            'status': 'success',
            'session_id': session_id,
            'detections': {
                'face_detected': bool(detection_results.get('face_detected', False)),
                'person_count': int(detection_results.get('person_count', 0)),
                'is_blurred': bool(detection_results.get('is_blurred', False)),
                'head_pose': detection_results.get('head_pose'),
                'head_pose_violations': detection_results.get('head_pose_violations', {})
            },
            'violations': violations,
            'violation_count': int(session['violation_count']),
            'auto_actions': auto_actions,
            'timestamp': current_time.isoformat()
        }
        
        # Log response structure for debugging - ALWAYS log this to help debug
        logger.info(
            f"Response for session {session_id}: "
            f"detections.person_count={response['detections']['person_count']}, "
            f"detections.face_detected={response['detections']['face_detected']}, "
            f"violations_count={len(response['violations'])}, "
            f"has_detections={'detections' in response}"
        )
        
        return response
    
    async def process_event(
        self,
        event_type: str,
        event_data: Dict,
        session_id: str
    ) -> Dict:
        """
        Process frontend events (tab switch, copy/paste, etc.)
        
        Args:
            event_type: Type of event (tab_switch, copy, paste, focus_loss, etc.)
            event_data: Event-specific data
            session_id: Session identifier
            
        Returns:
            Event processing result
        """
        session = self.sessions[session_id]
        current_time = datetime.now()
        
        violation = {
            'type': f'frontend_event_{event_type}',
            'timestamp': current_time.isoformat(),
            'severity': 'high' if event_type in ['tab_switch', 'focus_loss'] else 'medium',
            'details': event_data
        }
        
        self._add_violation(session_id, violation)
        
        # Check for auto-actions
        auto_actions = self._check_auto_actions(session_id)
        
        # Include detections in event response (use last known state)
        last_face_detected = session.get('last_face_detected', False)
        
        return {
            'status': 'success',
            'session_id': session_id,
            'event_processed': True,
            'violation': violation,
            'violation_count': int(session['violation_count']),
            'auto_actions': auto_actions,
            'detections': {
                'face_detected': bool(last_face_detected),
                'person_count': 0,  # Events don't process frames, so use 0
                'is_blurred': False,
                'head_pose': None,
                'head_pose_violations': {}
            },
            'violations': [violation],  # Include violation in violations array for consistency
            'timestamp': current_time.isoformat()
        }
    
    def _add_violation(self, session_id: str, violation: Dict):
        """Add violation to session"""
        session = self.sessions[session_id]
        session['violations'].append(violation)
        session['violation_count'] += 1
        
        logger.info(
            f"Violation added to session {session_id}: {violation['type']} "
            f"(Total: {session['violation_count']})"
        )
    
    def _check_auto_actions(self, session_id: str) -> Dict:
        """
        Check if auto-flag or auto-submit should be triggered
        
        Returns:
            Dict with auto-action flags
        """
        session = self.sessions[session_id]
        count = session['violation_count']
        
        auto_actions = {
            'auto_flag': False,
            'auto_submit': False
        }
        
        # Only trigger auto-submit once
        if count >= self.settings.AUTO_SUBMIT_THRESHOLD and not session.get('auto_submit_triggered', False):
            auto_actions['auto_submit'] = True
            session['auto_submit_triggered'] = True
            logger.warning(f"Auto-submit triggered for session {session_id} (violation count: {count})")
        # Only trigger auto-flag once (and only if auto-submit hasn't been triggered)
        elif count >= self.settings.AUTO_FLAG_THRESHOLD and not session.get('auto_flag_triggered', False) and not session.get('auto_submit_triggered', False):
            auto_actions['auto_flag'] = True
            session['auto_flag_triggered'] = True
            logger.warning(f"Auto-flag triggered for session {session_id} (violation count: {count})")
        
        return auto_actions
    
    def get_session_status(self, session_id: str) -> Dict:
        """Get current status of a session"""
        if session_id not in self.sessions:
            return {'error': 'Session not found'}
        
        session = self.sessions[session_id]
        return {
            'session_id': session_id,
            'violation_count': session['violation_count'],
            'violations': session['violations'][-10:],  # Last 10 violations
            'state': session['state']
        }
    
    def reset_session(self, session_id: str):
        """Reset a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Session {session_id} reset")
    
    def cleanup(self):
        """Cleanup all resources"""
        self.person_detector.cleanup()
        self.face_detector.cleanup()
        logger.info("Violation engine cleaned up")
