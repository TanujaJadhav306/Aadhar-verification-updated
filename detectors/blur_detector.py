"""
Blur Detection Module
Detects if webcam is blurred or covered using Laplacian variance
"""

import cv2
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class BlurDetector:
    """Laplacian variance-based blur detection"""
    
    def __init__(self, threshold: float = 100.0):
        """
        Initialize blur detector
        
        Args:
            threshold: Laplacian variance threshold below which frame is considered blurred
        """
        self.threshold = threshold
    
    def detect_blur(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if frame is blurred
        
        Args:
            frame: Input frame (BGR or grayscale)
            
        Returns:
            Tuple of (is_blurred, variance_score)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Additional validation: Check if frame is actually dark (covered) vs just low detail
        mean_brightness = np.mean(gray)
        
        # Only consider blurred if:
        # 1. Variance is below threshold AND
        # 2. Frame is actually dark (covered camera) OR variance is extremely low
        # This prevents false positives from normal low-detail frames
        is_blurred = variance < self.threshold and (mean_brightness < 50 or variance < 20)
        
        logger.debug(f"Blur detection: variance={variance:.2f}, brightness={mean_brightness:.2f}, is_blurred={is_blurred}")
        
        return is_blurred, variance
    
    def is_covered(self, frame: np.ndarray) -> bool:
        """
        Detect if camera is covered (very low variance + dark)
        
        Args:
            frame: Input frame
            
        Returns:
            True if camera appears covered
        """
        is_blurred, variance = self.detect_blur(frame)
        
        # Convert to grayscale for brightness check
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Check if frame is very dark (covered camera)
        mean_brightness = np.mean(gray)
        
        # Covered if very low variance AND very dark
        return is_blurred and mean_brightness < 30
