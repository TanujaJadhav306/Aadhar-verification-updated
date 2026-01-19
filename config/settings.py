"""
Configuration settings for the proctoring service
Uses environment variables with sensible defaults
"""

from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Union
import os
import json


class Settings(BaseSettings):
    """Application settings"""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS settings - can be JSON array or comma-separated string
    CORS_ORIGINS: Union[str, List[str]] = "http://localhost:3000,http://localhost:5173"
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS from string (comma-separated or JSON) to list"""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            # Try parsing as JSON first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
            # If not JSON, treat as comma-separated string
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            return origins if origins else ["http://localhost:3000", "http://localhost:5173"]
        return ["http://localhost:3000", "http://localhost:5173"]
    
    # Detection thresholds
    MULTIPLE_PERSON_VIOLATION_SECONDS: float = 3.0
    FACE_MISSING_VIOLATION_SECONDS: float = 5.0
    HEAD_POSE_VIOLATION_SECONDS: float = 3.0
    BLUR_VIOLATION_SECONDS: float = 5.0
    
    # Head pose thresholds (in degrees)
    HEAD_DOWN_THRESHOLD: float = 25.0
    HEAD_LEFT_THRESHOLD: float = 30.0
    HEAD_RIGHT_THRESHOLD: float = 30.0
    HEAD_AWAY_THRESHOLD: float = 45.0
    
    # Blur detection threshold (increased to reduce false positives)
    # Lower values = more sensitive (more false positives)
    # Higher values = less sensitive (may miss actual blur)
    BLUR_THRESHOLD: float = 50.0  # Reduced from 100.0 to be less sensitive
    
    # Violation limits
    MAX_VIOLATIONS: int = 10
    AUTO_FLAG_THRESHOLD: int = 5
    AUTO_SUBMIT_THRESHOLD: int = 10
    
    # YOLO model settings
    YOLO_MODEL_PATH: str = "models/yolov8n.pt"  # YOLOv8 nano model
    YOLO_CONFIDENCE: float = 0.5
    
    # MediaPipe settings
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # Frame processing
    FRAME_PROCESSING_INTERVAL: int = 1  # Process every Nth frame
    
    class Config:
        env_file = ".env"
        case_sensitive = True


_settings = None


def get_settings() -> Settings:
    """Get settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
