"""
Simple test client for the proctoring service
Demonstrates how to send frames and events
"""

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
from datetime import datetime


async def test_websocket():
    """Test WebSocket connection and frame sending"""
    uri = "ws://localhost:8000/ws/proctoring/test-session-123"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Test 1: Send a test event
            print("\n[Test 1] Sending tab switch event...")
            event_message = {
                "type": "event",
                "event_type": "tab_switch",
                "event_data": {
                    "timestamp": datetime.now().isoformat(),
                    "url": "https://example.com"
                }
            }
            await websocket.send(json.dumps(event_message))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Test 2: Create a test frame (simple colored image)
            print("\n[Test 2] Creating and sending test frame...")
            test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            test_frame[:] = (100, 150, 200)  # Blue-ish color
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', test_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            frame_message = {
                "type": "frame",
                "frame": frame_base64,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "width": 640,
                    "height": 480
                }
            }
            await websocket.send(json.dumps(frame_message))
            
            response = await websocket.recv()
            result = json.loads(response)
            print(f"Response: {json.dumps(result, indent=2)}")
            
            # Test 3: Send multiple frames to test violation tracking
            print("\n[Test 3] Sending multiple frames...")
            for i in range(3):
                frame_message = {
                    "type": "frame",
                    "frame": frame_base64,
                    "metadata": {"frame_number": i + 1}
                }
                await websocket.send(json.dumps(frame_message))
                response = await websocket.recv()
                result = json.loads(response)
                print(f"Frame {i+1} - Violations: {len(result.get('violations', []))}, "
                      f"Count: {result.get('violation_count', 0)}")
                await asyncio.sleep(0.5)
            
            print("\n✓ All tests completed!")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure the server is running: python main.py")


async def test_rest_api():
    """Test REST API endpoints"""
    import aiohttp
    
    base_url = "http://localhost:8000/api/v1"
    
    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        print("\n[Test] Health check...")
        async with session.get(f"{base_url}/health") as response:
            result = await response.json()
            print(f"Health: {json.dumps(result, indent=2)}")
        
        # Test session status
        print("\n[Test] Get session status...")
        async with session.get(f"{base_url}/proctoring/session/test-session-123") as response:
            result = await response.json()
            print(f"Session status: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    print("=" * 50)
    print("Proctoring Service Test Client")
    print("=" * 50)
    print("\nMake sure the server is running: python main.py")
    print("Press Ctrl+C to exit\n")
    
    try:
        # Test WebSocket
        asyncio.run(test_websocket())
        
        # Test REST API
        # asyncio.run(test_rest_api())
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
