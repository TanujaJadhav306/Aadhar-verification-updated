"""
YOLO-based Person Detection Module
Detects multiple people in the frame
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PersonDetector:
    """YOLO-based person detection"""
    
    def __init__(self, model_path: str, confidence: float = 0.5):
        """
        Initialize YOLO person detector
        
        Args:
            model_path: Path to YOLO model file
            confidence: Minimum confidence threshold
        """
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            logger.info(f"Attempting to load YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully from {self.model_path}")
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            logger.warning("Person detection will be disabled - install ultralytics to enable")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            logger.warning("Person detection will be disabled due to model loading error")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> Tuple[int, List[dict]]:
        """
        Detect persons in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (person_count, detections)
            detections: List of dicts with 'bbox', 'confidence', 'class'
        """
        if self.model is None:
            logger.warning("❌ YOLO model not loaded - person detection disabled")
            return 0, []
        
        try:
            # Validate frame
            if frame is None or frame.size == 0:
                logger.error("❌ Invalid frame provided to person detector")
                logger.error(f"   Frame: {frame}")
                logger.error(f"   Frame size: {frame.size if frame is not None else 'N/A'}")
                return 0, []
            
            logger.info(f"🔍 Running YOLO detection")
            logger.info(f"   Frame shape: {frame.shape}")
            logger.info(f"   Frame dtype: {frame.dtype}")
            logger.info(f"   Confidence threshold: {self.confidence}")
            logger.info(f"   Model loaded: {self.model is not None}")
            
            # Run YOLO inference
            logger.info("   Running YOLO inference...")
            results = self.model(frame, conf=self.confidence, verbose=False)
            logger.info(f"   YOLO inference complete, results type: {type(results)}")
            
            person_count = 0
            detections = []
            
            # Process results
            logger.info(f"   Processing YOLO results...")
            for idx, result in enumerate(results):
                logger.info(f"   Result {idx}: has boxes: {result.boxes is not None}")
                if result.boxes is not None:
                    logger.info(f"   Result {idx}: boxes length: {len(result.boxes)}")
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    logger.info(f"   Processing {len(boxes)} boxes...")
                    for box_idx, box in enumerate(boxes):
                        box_cls = int(box.cls)
                        logger.debug(f"   Box {box_idx}: class={box_cls}, conf={float(box.conf[0].cpu().numpy()):.2f}")
                        # Class 0 in COCO dataset is 'person'
                        if box_cls == 0:  # Person class
                            person_count += 1
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': conf,
                                'class': 'person'
                            })
                            logger.info(f"   ✅ Person {person_count} detected: bbox={[int(x1), int(y1), int(x2), int(y2)]}, conf={conf:.2f}")
            
            if person_count > 0:
                confidences = [f"{d['confidence']:.2f}" for d in detections]
                logger.info(f"✅ YOLO detected {person_count} person(s) with confidences: {confidences}")
            else:
                logger.info(f"❌ YOLO detected 0 persons in frame")
                logger.info(f"   Frame shape: {frame.shape}")
                logger.info(f"   Total boxes found: {sum(len(r.boxes) if r.boxes is not None else 0 for r in results)}")
            
            return person_count, detections
            
        except Exception as e:
            logger.error(f"Error in person detection: {e}", exc_info=True)
            return 0, []
    
    def has_multiple_persons(self, frame: np.ndarray) -> bool:
        """
        Check if multiple persons are detected
        
        Args:
            frame: Input frame
            
        Returns:
            True if more than 1 person detected
        """
        person_count, _ = self.detect(frame)
        return person_count > 1
    
    def cleanup(self):
        """Cleanup resources"""
        self.model = None
