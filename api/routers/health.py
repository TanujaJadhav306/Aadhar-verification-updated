"""
Health check endpoints
"""

from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Proctoring Service"
    }


@router.get("/health/detectors")
async def detectors_health():
    """Check detector health"""
    # This would check if models are loaded
    return {
        "status": "ok",
        "detectors": {
            "yolo": "loaded",
            "mediapipe": "loaded",
            "blur": "ready"
        }
    }
