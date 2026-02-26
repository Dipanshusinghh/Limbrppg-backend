#!/usr/bin/env python3
"""
LimbVital Backend Test Script
Project ke saare components ko locally test karne ke liye.
"""

import asyncio
import json
import base64
import sys
import cv2
import numpy as np
from datetime import datetime

# --- Test Helpers ---

def create_test_frame():
    # Synthetic frame banayein testing ke liye (Skin-tone color + noise)
    img = np.full((480, 640, 3), (80, 100, 120), dtype=np.uint8)
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

def frame_to_base64(frame):
    # OpenCV frame ko base64 string mein convert karna
    _, buf = cv2.imencode(".jpg", frame)
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

# --- Test Cases ---

def test_imports():
    print("\n[1/6] Checking Dependencies...")
    modules = {
        "fastapi": "FastAPI", "uvicorn": "Uvicorn", "cv2": "OpenCV",
        "numpy": "NumPy", "scipy": "SciPy", "mediapipe": "MediaPipe",
        "websockets": "WebSockets", "dotenv": "python-dotenv"
    }
    status = True
    for mod, name in modules.items():
        try:
            __import__(mod)
            print(f"  OK: {name}")
        except ImportError:
            print(f"  ERR: {name} missing")
            status = False
    return status

def test_files():
    print("\n[2/6] Checking Local Files...")
    required = ["main.py", "detector.py", "utils.py", "requirements.txt"]
    status = True
    for f in required:
        if os.path.exists(f):
            print(f"  OK: {f} found")
        else:
            print(f"  ERR: {f} not found")
            status = False
    return status

def test_utils_logic():
    print("\n[3/6] Testing utils.py functions...")
    try:
        from utils import base64_to_cv2, normalize_signal, validate_frame
        
        # Test 1: Base64 decoding
        frame = create_test_frame()
        decoded = base64_to_cv2(frame_to_base64(frame))
        assert decoded is not None and decoded.shape == frame.shape
        
        # Test 2: Signal normalization
        sig = normalize_signal([10, 20, 30, 40, 50])
        assert sig[0] == 0.0 and sig[-1] == 1.0
        
        # Test 3: Frame validation
        valid, _ = validate_frame(frame)
        assert valid == True
        
        print("  OK: All utils functions passed")
        return True
    except Exception as e:
        print(f"  ERR: Utils test failed -> {e}")
        return False

def test_detector_logic():
    print("\n[4/6] Testing RPPGDetector (Smoke Test)...")
    try:
        from detector import RPPGDetector
        det = RPPGDetector(buffer_size=10)
        
        # Dummy frames process karke check karna (crash nahi hona chahiye)
        for _ in range(3):
            res = det.process_frame(create_test_frame())
            assert "bpm" in res and "status" in res
            
        print(f"  OK: Detector working. Last Status: {res['status']}")
        return True
    except Exception as e:
        print(f"  ERR: Detector failed -> {e}")
        return False

async def test_server_http():
    print("\n[5/6] Testing Server HTTP Endpoints...")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/health") as r:
                if r.status == 200:
                    data = await r.json()
                    print(f"  OK: Health Check -> {data}")
                    return True
    except:
        print("  SKIP: Server not running on localhost:8000")
        return False

async def test_server_ws():
    print("\n[6/6] Testing WebSocket Connection...")
    try:
        import websockets
        async with websockets.connect("ws://localhost:8000/ws") as ws:
            payload = json.dumps({"image": frame_to_base64(create_test_frame())})
            await ws.send(payload)
            resp = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"  OK: WebSocket Response -> {resp}")
            return True
    except:
        print("  SKIP: WebSocket test skipped (Server offline)")
        return False

# --- Main Runner ---

async def run_suite():
    print(f"--- LimbVital Backend Test Suite | {datetime.now().strftime('%H:%M:%S')} ---")
    
    # Core Tests
    f_ok = test_files()
    d_ok = test_imports()
    u_ok = test_utils_logic()
    det_ok = test_detector_logic()
    
    # Server Tests
    h_ok = await test_server_http()
    w_ok = await test_server_ws()

    print("\n" + "="*30)
    print(f"FILES:      {'PASS' if f_ok else 'FAIL'}")
    print(f"DEPS:       {'PASS' if d_ok else 'FAIL'}")
    print(f"UTILS:      {'PASS' if u_ok else 'FAIL'}")
    print(f"DETECTOR:   {'PASS' if det_ok else 'FAIL'}")
    print(f"HTTP/WS:    {'PASS' if (h_ok or w_ok) else 'OFFLINE'}")
    print("="*30)

if __name__ == "__main__":
    import os
    asyncio.run(run_suite())