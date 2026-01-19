# AI Proctoring Service

A production-ready AI-based online proctoring system built with FastAPI, OpenCV, MediaPipe, and YOLOv8. This service provides real-time detection of multiple persons, face movements, head pose estimation, and blur detection.

## Features

### Detection Capabilities

- ✅ **Multiple Person Detection**: Uses YOLOv8 to detect if more than one person is present in the frame
- ✅ **Face Detection & Tracking**: MediaPipe-based face detection and tracking
- ✅ **Head Pose Estimation**: Detects looking down, left, right, or away from screen
- ✅ **Eye Gaze Approximation**: Estimates attention score based on head pose
- ✅ **Blur Detection**: Laplacian variance-based blur and camera coverage detection
- ✅ **Frontend Event Monitoring**: Receives and processes tab switches, copy/paste, focus loss events
- ✅ **AI Tool Detection Support**: Framework for detecting AI-enabled tools (client-side integration required)
- ✅ **Violation Engine**: Tracks violations with configurable thresholds and auto-flag/auto-submit

### Architecture

```
Client (JS/WebRTC)
      |
      v
FastAPI Backend
      |
      ├── Person Detector (YOLOv8)
      ├── Face Detector (MediaPipe)
      ├── Head Pose Estimator
      ├── Blur Detector
      └── Violation Engine
```

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (optional, for faster YOLO inference)
- Webcam access

### Setup

1. **Clone or navigate to the proctoring-service directory**

```bash
cd proctoring-service
```

2. **Create a virtual environment**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download YOLO model (automatic on first run)**

The YOLOv8 model will be automatically downloaded on first use. Alternatively, you can download it manually:

```bash
mkdir -p models
# The model will be downloaded automatically when PersonDetector is initialized
```

5. **Configure environment variables**

```bash
cp .env.example .env
# Edit .env with your settings
```

## Configuration

All configuration is done via environment variables. See `.env.example` for all available options.

### Key Configuration Options

- **Detection Thresholds**: Time in seconds before a violation is triggered
- **Head Pose Thresholds**: Angle thresholds in degrees for head pose violations
- **Blur Threshold**: Laplacian variance threshold (lower = more sensitive)
- **Violation Limits**: Auto-flag and auto-submit thresholds

## Usage

### Start the Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### WebSocket Connection

For real-time proctoring, connect via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/proctoring/session-123');

// Send frame
ws.send(JSON.stringify({
  type: 'frame',
  frame: base64EncodedFrame,
  metadata: { timestamp: Date.now() }
}));

// Send event
ws.send(JSON.stringify({
  type: 'event',
  event_type: 'tab_switch',
  event_data: { timestamp: Date.now() }
}));
```

### REST API Endpoints

#### Process Frame (POST)
```bash
POST /api/v1/proctoring/frame
Content-Type: application/json

{
  "frame": "base64_encoded_image",
  "session_id": "session-123",
  "metadata": {}
}
```

#### Process Event (POST)
```bash
POST /api/v1/proctoring/event
Content-Type: application/json

{
  "event_type": "tab_switch",
  "event_data": {},
  "session_id": "session-123"
}
```

#### Get Session Status (GET)
```bash
GET /api/v1/proctoring/session/{session_id}
```

#### Reset Session (DELETE)
```bash
DELETE /api/v1/proctoring/session/{session_id}
```

## Project Structure

```
proctoring-service/
├── main.py                 # FastAPI application entry point
├── config/
│   ├── __init__.py
│   └── settings.py         # Configuration management
├── detectors/
│   ├── __init__.py
│   ├── person_detector.py  # YOLO person detection
│   ├── face_detector.py    # MediaPipe face & head pose
│   └── blur_detector.py    # Blur detection
├── services/
│   ├── __init__.py
│   ├── violation_engine.py # Main violation tracking engine
│   └── websocket_manager.py # WebSocket connection manager
├── api/
│   └── routers/
│       ├── __init__.py
│       ├── health.py       # Health check endpoints
│       └── proctoring.py   # Proctoring API endpoints
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Detection Strategies

### Multiple Person Detection
- Uses YOLOv8 person detection
- Violation triggered if >1 person detected for configured duration
- Configurable via `MULTIPLE_PERSON_VIOLATION_SECONDS`

### Face Movements
- Head pose estimation using MediaPipe Face Mesh
- Detects: looking down, left, right, away
- Violation triggered if sustained movement exceeds threshold
- Configurable via `HEAD_POSE_VIOLATION_SECONDS` and angle thresholds

### Face Missing
- Detects when face is not visible in frame
- Violation triggered after configured duration
- Configurable via `FACE_MISSING_VIOLATION_SECONDS`

### Blur Detection
- Uses Laplacian variance to detect blur
- Also detects covered camera (low variance + dark)
- Configurable via `BLUR_THRESHOLD` and `BLUR_VIOLATION_SECONDS`

## Frontend Integration

### Sending Frames

Capture webcam frames and send as base64:

```javascript
const video = document.getElementById('webcam');
const canvas = document.createElement('canvas');
canvas.width = video.videoWidth;
canvas.height = video.videoHeight;
const ctx = canvas.getContext('2d');
ctx.drawImage(video, 0, 0);
const frameData = canvas.toDataURL('image/jpeg', 0.8);
```

### Monitoring Frontend Events

```javascript
// Tab switch detection
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    sendEvent('tab_switch', {});
  }
});

// Copy/paste detection
document.addEventListener('copy', (e) => {
  sendEvent('copy', {});
});

document.addEventListener('paste', (e) => {
  sendEvent('paste', {});
});

// Focus loss
window.addEventListener('blur', () => {
  sendEvent('focus_loss', {});
});
```

### AI Tool Detection (Client-Side)

AI tool detection should be implemented on the client side and reported as events:

```javascript
// Detect AI tools on screen (requires browser extension or desktop app)
function detectAITools() {
  // Method 1: Monitor clipboard for AI-generated content patterns
  document.addEventListener('paste', (e) => {
    const pastedText = e.clipboardData.getData('text');
    if (isAIGenerated(pastedText)) {
      sendEvent('ai_tool_detected', {
        type: 'clipboard_ai',
        content: pastedText.substring(0, 100) // First 100 chars
      });
    }
  });
  
  // Method 2: Monitor for AI tool windows (requires desktop app)
  // This would require Electron or similar to access system processes
  // Check for processes like: ChatGPT, Claude, Copilot, etc.
  
  // Method 3: Screen content analysis (send screenshots to backend)
  // The backend can analyze screenshots for AI tool interfaces
}

// Detect AI tools in system (requires desktop app/extension)
// Example for Electron app:
const { exec } = require('child_process');

function checkAITools() {
  // Windows
  exec('tasklist', (error, stdout) => {
    const aiTools = ['ChatGPT', 'Claude', 'Copilot', 'Bard'];
    aiTools.forEach(tool => {
      if (stdout.includes(tool)) {
        sendEvent('ai_tool_detected', {
          type: 'process',
          tool: tool,
          timestamp: Date.now()
        });
      }
    });
  });
  
  // Linux/Mac alternative: ps aux | grep -i "chatgpt|claude"
}
```

**Note**: Full AI tool detection requires:
- Browser extension for web-based detection
- Desktop application (Electron) for system-level detection
- Screen content analysis via screenshot analysis (can be integrated with frame processing)

## Performance Optimization

- **Frame Processing Interval**: Set `FRAME_PROCESSING_INTERVAL` to process every Nth frame
- **GPU Acceleration**: YOLO will automatically use GPU if available
- **Model Selection**: Use YOLOv8n (nano) for faster inference, YOLOv8s/m/l for better accuracy

## Troubleshooting

### YOLO Model Not Found
- The model will be automatically downloaded on first use
- Ensure internet connection for first run
- Or manually download from Ultralytics and place in `models/` directory

### MediaPipe Issues
- Ensure OpenCV is properly installed: `pip install opencv-python`
- Check camera permissions

### Performance Issues
- Reduce `FRAME_PROCESSING_INTERVAL` to process fewer frames
- Use smaller YOLO model (yolov8n.pt)
- Enable GPU acceleration if available

## License

This project is part of the Bravens proctoring system.

## Support

For issues or questions, please refer to the main project documentation.
