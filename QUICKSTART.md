# Quick Start Guide

## Installation (5 minutes)

### 1. Navigate to the service directory
```bash
cd proctoring-service
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Note**: This will download:
- FastAPI and server dependencies (~50MB)
- OpenCV and MediaPipe (~200MB)
- PyTorch and YOLOv8 (~1.5GB for CPU, ~2GB for GPU)

### 4. Configure environment
```bash
# Copy example env file
cp .env.example .env

# Edit .env with your settings (optional - defaults work)
```

### 5. Start the server
```bash
python main.py
```

The service will be available at `http://localhost:8000`

## Testing the Service

### 1. Check Health
```bash
curl http://localhost:8000/api/v1/health
```

### 2. View API Documentation
Open in browser: `http://localhost:8000/docs`

### 3. Test with WebSocket (JavaScript)

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws/proctoring/test-session-1');

ws.onopen = () => {
    console.log('Connected');
    
    // Get webcam stream
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            const video = document.createElement('video');
            video.srcObject = stream;
            video.play();
            
            // Capture and send frames
            setInterval(() => {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                
                const frameData = canvas.toDataURL('image/jpeg', 0.8);
                
                ws.send(JSON.stringify({
                    type: 'frame',
                    frame: frameData,
                    metadata: { timestamp: Date.now() }
                }));
            }, 1000); // Send every second
        });
};

ws.onmessage = (event) => {
    const result = JSON.parse(event.data);
    console.log('Detection result:', result);
    
    if (result.violations && result.violations.length > 0) {
        console.warn('Violations detected:', result.violations);
    }
    
    if (result.auto_actions.auto_flag) {
        console.error('AUTO-FLAG triggered!');
    }
    
    if (result.auto_actions.auto_submit) {
        console.error('AUTO-SUBMIT triggered!');
    }
};
```

## Configuration Quick Reference

### Key Settings in `.env`

```env
# How long before violation triggers (seconds)
MULTIPLE_PERSON_VIOLATION_SECONDS=3.0
FACE_MISSING_VIOLATION_SECONDS=5.0
HEAD_POSE_VIOLATION_SECONDS=3.0

# Head pose angles (degrees)
HEAD_DOWN_THRESHOLD=25.0      # Looking down
HEAD_LEFT_THRESHOLD=30.0      # Looking left
HEAD_RIGHT_THRESHOLD=30.0     # Looking right
HEAD_AWAY_THRESHOLD=45.0      # Looking away

# Violation limits
AUTO_FLAG_THRESHOLD=5         # Flag after 5 violations
AUTO_SUBMIT_THRESHOLD=10      # Auto-submit after 10 violations
```

## Common Issues

### YOLO Model Download
- First run will download YOLOv8n model (~6MB)
- Requires internet connection
- Model saved to `models/yolov8n.pt`

### GPU Not Detected
- YOLO will use CPU if GPU not available
- Install CUDA for GPU acceleration
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

### MediaPipe Issues
- Ensure OpenCV is installed: `pip install opencv-python`
- Check camera permissions

### Port Already in Use
- Change PORT in `.env` file
- Or kill process: `lsof -ti:8000 | xargs kill` (Linux/Mac)

## Next Steps

1. **Integrate with Frontend**: See README.md for frontend integration examples
2. **Customize Thresholds**: Adjust `.env` based on your requirements
3. **Monitor Performance**: Check logs for detection performance
4. **Scale**: Use multiple workers for production: `uvicorn main:app --workers 4`

## Support

For detailed documentation, see `README.md`
