#!/usr/bin/env python3
"""
LimbVital Backend Test Script
Tests core functionality without requiring a running server for unit tests.
"""

import asyncio
import json
import base64
import sys
import cv2
import numpy as np
from datetime import datetime


# ── Helpers ──────────────────────────────────────────────────────────────────

def create_test_frame() -> np.ndarray:
    """Create a synthetic BGR frame with a skin-tone-ish colour and noise."""
    img = np.full((480, 640, 3), (80, 100, 120), dtype=np.uint8)  # BGR
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def frame_to_base64(frame: np.ndarray) -> str:
    """Encode an OpenCV frame as a JPEG data-URL."""
    _, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_imports() -> bool:
    print("\n🔍 Testing Dependencies…")
    modules = {
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "scipy": "SciPy",
        "mediapipe": "MediaPipe",
        "websockets": "WebSockets",
        "aiohttp": "aiohttp",
        "dotenv": "python-dotenv",
    }
    ok = True
    for mod, name in modules.items():
        try:
            __import__(mod)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} not installed")
            ok = False
    return ok


def test_local_files() -> bool:
    """Check that all required project files exist."""
    print("\n🔍 Checking Local Files…")
    required = ["main.py", "detector.py", "utils.py", "requirements.txt", ".env"]
    ok = True
    for f in required:
        try:
            open(f).close()
            print(f"  ✅ {f}")
        except FileNotFoundError:
            print(f"  ❌ {f} missing")
            ok = False
    return ok


def test_utils() -> bool:
    """Unit-test utils.py without a server."""
    print("\n🔍 Testing utils.py…")
    try:
        from utils import base64_to_cv2, normalize_signal, validate_frame

        frame = create_test_frame()
        b64 = frame_to_base64(frame)
        decoded = base64_to_cv2(b64)
        assert decoded is not None and decoded.shape == frame.shape, "base64_to_cv2 failed"
        print("  ✅ base64_to_cv2")

        sig = normalize_signal([10, 20, 30, 40, 50])
        assert abs(float(sig[0])) < 1e-9 and abs(float(sig[-1]) - 1.0) < 1e-9
        print("  ✅ normalize_signal")

        valid, msg = validate_frame(frame)
        assert valid, msg
        valid2, _ = validate_frame(None)
        assert not valid2
        print("  ✅ validate_frame")

        return True
    except Exception as e:
        print(f"  ❌ utils test failed: {e}")
        return False


def test_detector_local() -> bool:
    """
    Smoke-test RPPGDetector with synthetic frames (no real face expected).
    Just checks it doesn't crash and returns the expected dict shape.
    """
    print("\n🔍 Testing detector.py (local / no server)…")
    try:
        from detector import RPPGDetector
        det = RPPGDetector(buffer_size=10)

        for _ in range(3):
            frame = create_test_frame()
            result = det.process_frame(frame)
            assert "bpm" in result and "spo2" in result and "status" in result, \
                f"Unexpected result shape: {result}"

        print("  ✅ RPPGDetector smoke test passed")
        print(f"     Last result: {result}")
        return True
    except Exception as e:
        print(f"  ❌ Detector test failed: {e}")
        return False


async def test_http() -> bool:
    """Test /  and /health HTTP endpoints (server must be running)."""
    try:
        import aiohttp
    except ImportError:
        print("  ⚠️  aiohttp not installed – skipping HTTP tests")
        return False

    print("\n🔍 Testing HTTP Endpoints…")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/") as r:
                assert r.status == 200
                data = await r.json()
                print(f"  ✅ GET /  → {data.get('status')}")

            async with session.get("http://localhost:8000/health") as r:
                assert r.status == 200
                data = await r.json()
                print(f"  ✅ GET /health → {data}")
        return True
    except aiohttp.ClientConnectorError:
        print("  ❌ Cannot connect – is the server running?  (python main.py)")
        return False
    except Exception as e:
        print(f"  ❌ HTTP test failed: {e}")
        return False


async def test_websocket() -> bool:
    """Send one frame over WebSocket and check the response (server must be running)."""
    try:
        import websockets
    except ImportError:
        print("  ⚠️  websockets not installed – skipping WS tests")
        return False

    print("\n🔍 Testing WebSocket /ws…")
    try:
        async with websockets.connect("ws://localhost:8000/ws") as ws:
            print("  ✅ WebSocket connected")

            payload = json.dumps({"image": frame_to_base64(create_test_frame())})
            await ws.send(payload)

            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            data = json.loads(raw)
            assert "bpm" in data and "status" in data
            print(f"  ✅ Response: {data}")
        return True
    except ConnectionRefusedError:
        print("  ❌ Connection refused – is the server running?  (python main.py)")
        return False
    except Exception as e:
        print(f"  ❌ WebSocket test failed: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("""
╔══════════════════════════════════════════════════╗
║       LimbVital Backend Test Suite               ║
╚══════════════════════════════════════════════════╝""")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    files_ok  = test_local_files()
    deps_ok   = test_imports()
    utils_ok  = test_utils()
    detect_ok = test_detector_local()

    # Server-dependent tests (skip gracefully if not running)
    http_ok = await test_http()
    ws_ok   = await test_websocket()

    print("\n" + "=" * 52)
    print("📋 TEST SUMMARY")
    print("=" * 52)
    for label, result in [
        ("Files",      files_ok),
        ("Deps",       deps_ok),
        ("Utils",      utils_ok),
        ("Detector",   detect_ok),
        ("HTTP",       http_ok),
        ("WebSocket",  ws_ok),
    ]:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {label:<12} {status}")
    print("=" * 52)

    core_ok = all([files_ok, deps_ok, utils_ok, detect_ok])
    if core_ok:
        print("\n🎉 Core tests passed!")
        if not (http_ok and ws_ok):
            print("⚠️  Server tests skipped/failed – start the server and re-run.")
    else:
        print("\n⚠️  Some core tests failed – see errors above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")