# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client (Frontend)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Webcam     │  │   Screen     │  │   Events     │     │
│  │   Capture    │  │   Monitor    │  │   Monitor    │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           │                                 │
│                    WebSocket / REST API                      │
└───────────────────────────┼─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              FastAPI Proctoring Service                     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │           WebSocket Manager                         │    │
│  │  - Connection management                            │    │
│  │  - Session tracking                                │    │
│  └────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │           Violation Engine                          │    │
│  │  - Frame processing orchestration                   │    │
│  │  - Violation tracking                               │    │
│  │  - Auto-flag/auto-submit logic                      │    │
│  └──────┬───────────┬───────────┬───────────┬────────┘    │
│         │           │           │           │               │
│         ▼           ▼           ▼           ▼               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │  Person  │ │   Face   │ │   Blur   │ │  Events  │     │
│  │ Detector │ │ Detector │ │ Detector │ │ Handler  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
│     (YOLO)    (MediaPipe)   (OpenCV)                       │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Detectors

#### Person Detector (`detectors/person_detector.py`)
- **Technology**: YOLOv8 (Ultralytics)
- **Purpose**: Detect multiple persons in frame
- **Output**: Person count, bounding boxes, confidence scores
- **Violation**: Triggered when >1 person detected for configured duration

#### Face Detector (`detectors/face_detector.py`)
- **Technology**: MediaPipe Face Mesh
- **Purpose**: 
  - Face detection and tracking
  - Head pose estimation (pitch, yaw, roll)
  - Eye gaze approximation
- **Output**: Face landmarks, head pose angles, attention score
- **Violations**: 
  - Face missing
  - Looking down/left/right/away (sustained)

#### Blur Detector (`detectors/blur_detector.py`)
- **Technology**: OpenCV Laplacian variance
- **Purpose**: Detect blurred or covered camera
- **Output**: Blur status, variance score
- **Violation**: Triggered when blur detected for configured duration

### 2. Services

#### Violation Engine (`services/violation_engine.py`)
- **Responsibilities**:
  - Orchestrates all detectors
  - Tracks violation state per session
  - Manages violation thresholds and timing
  - Implements auto-flag/auto-submit logic
- **State Management**: Per-session state tracking with timestamps
- **Violation Types**:
  - `multiple_persons`: More than 1 person detected
  - `face_missing`: Face not detected
  - `head_pose_*`: Looking down/left/right/away
  - `blurred_camera`: Camera is blurred or covered
  - `frontend_event_*`: Tab switch, copy, paste, focus loss

#### WebSocket Manager (`services/websocket_manager.py`)
- **Responsibilities**:
  - Manage WebSocket connections
  - Session-to-connection mapping
  - Broadcast and personal messaging

### 3. API Layer

#### REST Endpoints (`api/routers/proctoring.py`)
- `POST /api/v1/proctoring/frame`: Process single frame
- `POST /api/v1/proctoring/event`: Process frontend event
- `GET /api/v1/proctoring/session/{session_id}`: Get session status
- `DELETE /api/v1/proctoring/session/{session_id}`: Reset session

#### WebSocket Endpoint (`main.py`)
- `WS /ws/proctoring/{session_id}`: Real-time bidirectional communication

### 4. Configuration (`config/settings.py`)

All settings configurable via environment variables:
- Detection thresholds (time-based)
- Head pose angle thresholds
- Blur detection threshold
- Violation limits
- Model paths and confidence levels

## Data Flow

### Frame Processing Flow

```
1. Client captures webcam frame
2. Frame encoded to base64
3. Sent via WebSocket/REST API
4. Violation Engine receives frame
5. Frame decoded to OpenCV format
6. Parallel processing:
   - Person Detector → YOLO inference
   - Face Detector → MediaPipe processing
   - Blur Detector → Laplacian variance
7. Results aggregated
8. Violation state updated
9. Response sent to client
```

### Event Processing Flow

```
1. Frontend event occurs (tab switch, copy, etc.)
2. Event sent via WebSocket/REST API
3. Violation Engine receives event
4. Violation created immediately
5. Violation count updated
6. Auto-actions checked
7. Response sent to client
```

## Session State Management

Each session maintains:
- **Violations**: List of all violations with timestamps
- **Violation Count**: Total count for auto-actions
- **State Tracking**:
  - `multiple_person_start`: When multiple persons first detected
  - `face_missing_start`: When face first went missing
  - `head_pose_violation_start`: When head pose violation started
  - `blur_start`: When blur first detected
  - `last_head_pose`: Last detected head pose violation type
  - `frame_count`: Total frames processed

## Violation Timing Logic

Violations are only triggered after sustained detection:
1. **Detection Phase**: Condition detected, start timer
2. **Sustained Phase**: Condition persists, timer continues
3. **Violation Phase**: Timer exceeds threshold, violation created
4. **Reset Phase**: Condition clears, timer resets

This prevents false positives from momentary detections.

## Auto-Actions

- **Auto-Flag**: Triggered when violation count >= `AUTO_FLAG_THRESHOLD`
- **Auto-Submit**: Triggered when violation count >= `AUTO_SUBMIT_THRESHOLD`

Both actions are included in response for client-side handling.

## Performance Considerations

1. **Frame Processing Interval**: Process every Nth frame to reduce load
2. **Model Selection**: YOLOv8n (nano) for speed, larger models for accuracy
3. **GPU Acceleration**: Automatic if CUDA available
4. **Async Processing**: All I/O operations are async
5. **Connection Pooling**: WebSocket connections managed efficiently

## Security Considerations

1. **Input Validation**: All inputs validated via Pydantic models
2. **Session Isolation**: Sessions are isolated, no cross-session data
3. **Error Handling**: Comprehensive error handling and logging
4. **CORS**: Configurable CORS for frontend integration
5. **Rate Limiting**: Can be added via middleware (not included)

## Scalability

- **Horizontal Scaling**: Stateless design allows multiple instances
- **Session Storage**: In-memory (can be moved to Redis for distributed systems)
- **Model Loading**: Models loaded once per instance
- **WebSocket**: Can use WebSocket load balancer for scaling

## Future Enhancements

1. **Database Integration**: Store violations in database
2. **Redis Session Storage**: For distributed deployments
3. **Model Optimization**: ONNX conversion for faster inference
4. **Advanced Gaze Tracking**: More sophisticated eye gaze estimation
5. **Screen Content Analysis**: Detect AI tools in screenshots
6. **Audio Monitoring**: Detect background conversations
7. **Machine Learning**: Train custom models for specific violations
