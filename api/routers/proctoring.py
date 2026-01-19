"""
Proctoring API endpoints
"""

from fastapi import APIRouter, HTTPException

router = APIRouter()

# Import violation engine (will be injected via dependency)
violation_engine = None


def set_violation_engine(engine):
    """Set violation engine instance"""
    global violation_engine
    violation_engine = engine


@router.get("/proctoring/session/{session_id}")
async def get_session_status(session_id: str):
    """Get current status of a proctoring session"""
    if violation_engine is None:
        raise HTTPException(status_code=503, detail="Violation engine not initialized")
    
    status = violation_engine.get_session_status(session_id)
    return status


@router.delete("/proctoring/session/{session_id}")
async def reset_session(session_id: str):
    """Reset a proctoring session"""
    if violation_engine is None:
        raise HTTPException(status_code=503, detail="Violation engine not initialized")
    
    violation_engine.reset_session(session_id)
    return {"status": "success", "message": f"Session {session_id} reset"}
