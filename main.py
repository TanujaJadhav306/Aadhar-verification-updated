"""
FastAPI Proctoring Service - Main Application
AI-based online proctoring system with real-time detection
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from config.settings import get_settings
from api.routers import proctoring, health, face_verification
from services.violation_engine import ViolationEngine
from services.websocket_manager import WebSocketManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Use INFO for production, DEBUG for detailed troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set specific loggers to INFO for detailed detection logs
logging.getLogger('services.violation_engine').setLevel(logging.INFO)
logging.getLogger('detectors.face_detector').setLevel(logging.INFO)
logging.getLogger('detectors.person_detector').setLevel(logging.INFO)

settings = get_settings()

# Global violation engine instance
violation_engine = None
websocket_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global violation_engine
    
    # Startup
    logger.info("Starting Proctoring Service...")
    violation_engine = ViolationEngine()
    logger.info("Violation Engine initialized")
    
    # Set violation engine in router
    proctoring.set_violation_engine(violation_engine)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Proctoring Service...")
    if violation_engine:
        violation_engine.cleanup()


# Create FastAPI app
app = FastAPI(
    title="AI Proctoring Service",
    description="AI-based online proctoring system with real-time detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(proctoring.router, prefix="/api/v1", tags=["Proctoring"])
app.include_router(face_verification.router, prefix="/api/v1", tags=["Face Verification"])




@app.websocket("/ws/proctoring/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time proctoring"""
    await websocket_manager.connect(websocket, session_id)
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_json()
            
            if data.get("type") == "frame":
                # Process frame through violation engine
                frame_data = data.get("frame", "")
                frame_data_length = len(frame_data) if frame_data else 0
                metadata = data.get('metadata', {})
                
                logger.info("=" * 80)
                logger.info(f"📥 FRAME RECEIVED - Session: {session_id}")
                logger.info(f"   Frame data length: {frame_data_length} bytes")
                logger.info(f"   Frame data prefix: {frame_data[:100] if frame_data else 'EMPTY'}...")
                logger.info(f"   Has data prefix: {frame_data.startswith('data:image') if frame_data else False}")
                logger.info(f"   Metadata: {metadata}")
                logger.info("=" * 80)
                
                try:
                    result = await violation_engine.process_frame(
                        frame_data=data.get("frame"),
                        session_id=session_id,
                        metadata=metadata
                    )
                    
                    logger.info(f"✅ FRAME PROCESSED - Session: {session_id}")
                    logger.info(f"   Result status: {result.get('status')}")
                    logger.info(f"   Has detections: {'detections' in result}")
                    if 'detections' in result:
                        det = result['detections']
                        logger.info(f"   face_detected: {det.get('face_detected')} (type: {type(det.get('face_detected'))})")
                        logger.info(f"   person_count: {det.get('person_count')} (type: {type(det.get('person_count'))})")
                        logger.info(f"   is_blurred: {det.get('is_blurred')}")
                    logger.info(f"   Violations count: {len(result.get('violations', []))}")
                    logger.info(f"   Total violation_count: {result.get('violation_count', 0)}")
                    logger.info("=" * 80)
                    
                    # Log result summary
                    if result.get("status") == "success":
                        violations_count = len(result.get("violations", []))
                        detections = result.get("detections", {})
                        face_detected = detections.get("face_detected")
                        person_count = detections.get("person_count", 0)
                        
                        if violations_count > 0:
                            logger.warning(f"Session {session_id}: {violations_count} violation(s) detected: {[v['type'] for v in result.get('violations', [])]} - SENDING TO FRONTEND")
                        
                        logger.info(f"Session {session_id}: face_detected={face_detected}, person_count={person_count}, violations={violations_count}")
                    elif result.get("status") == "skipped":
                        detections = result.get("detections", {})
                        logger.debug(f"Frame skipped for session {session_id}, detections={detections}")
                    else:
                        logger.warning(f"Frame processing error for session {session_id}: {result.get('message', 'Unknown error')}")
                    
                    # Ensure detections is always present in response
                    if "detections" not in result:
                        logger.warning(f"Response missing detections for session {session_id}, adding default")
                        result["detections"] = {
                            'face_detected': False,
                            'person_count': 0,
                            'is_blurred': False,
                            'head_pose': None,
                            'head_pose_violations': {}
                        }
                    
                    # Log the full response structure before sending (for debugging)
                    detections_in_result = result.get('detections', {})
                    logger.info(
                        f"Sending response to session {session_id}: "
                        f"status={result.get('status')}, "
                        f"has_detections={'detections' in result}, "
                        f"person_count={detections_in_result.get('person_count')}, "
                        f"face_detected={detections_in_result.get('face_detected')}, "
                        f"violations_count={len(result.get('violations', []))}"
                    )
                    
                    # Send result back to client
                    await websocket.send_json(result)
                except Exception as e:
                    logger.error(f"Error processing frame for session {session_id}: {e}", exc_info=True)
                    # Get session to use last known detection state
                    # Access sessions through the violation_engine's sessions dict
                    if hasattr(violation_engine, 'sessions') and session_id in violation_engine.sessions:
                        session = violation_engine.sessions[session_id]
                        last_face_detected = session.get('last_face_detected', False)
                        violation_count = session.get('violation_count', 0)
                    else:
                        last_face_detected = False
                        violation_count = 0
                    
                    await websocket.send_json({
                        'status': 'error',
                        'message': str(e),
                        'session_id': session_id,
                        'detections': {
                            'face_detected': bool(last_face_detected),
                            'person_count': 0,
                            'is_blurred': False,
                            'head_pose': None,
                            'head_pose_violations': {}
                        },
                        'violations': [],
                        'violation_count': int(violation_count),
                        'auto_actions': {'auto_flag': False, 'auto_submit': False},
                        'timestamp': datetime.now().isoformat()
                    })
            
            elif data.get("type") == "event":
                # Process frontend events (tab switch, copy/paste, etc.)
                result = await violation_engine.process_event(
                    event_type=data.get("event_type"),
                    event_data=data.get("event_data"),
                    session_id=session_id
                )
                
                # Ensure detections is always present in event response
                if "detections" not in result:
                    logger.warning(f"Event response missing detections for session {session_id}, adding default")
                    if hasattr(violation_engine, 'sessions') and session_id in violation_engine.sessions:
                        session = violation_engine.sessions[session_id]
                        last_face_detected = session.get('last_face_detected', False)
                    else:
                        last_face_detected = False
                    result["detections"] = {
                        'face_detected': bool(last_face_detected),
                        'person_count': 0,
                        'is_blurred': False,
                        'head_pose': None,
                        'head_pose_violations': {}
                    }
                
                await websocket.send_json(result)
    
    except WebSocketDisconnect:
        websocket_manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected for session: {session_id}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
