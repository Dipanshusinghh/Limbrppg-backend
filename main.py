"""
LimbVital rPPG Engine – FastAPI WebSocket Server
Run with:  python main.py
       or: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from detector import RPPGDetector
from utils import base64_to_cv2

load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 150))

app = FastAPI(title="LimbVital rPPG Engine", version="1.0.0")

# Allow all origins in dev; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound frame processing (keeps the event loop free)
executor = ThreadPoolExecutor(max_workers=4)

# Track active WebSocket connections for the health endpoint
active_connections: set[WebSocket] = set()


# ── HTTP endpoints ─────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "LimbVital rPPG Engine",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Health check used by monitoring tools and the test suite."""
    try:
        import mediapipe  # noqa: F401
        mediapipe_status = "operational"
    except ImportError:
        mediapipe_status = "not installed"

    return {
        "status": "healthy",
        "mediapipe": mediapipe_status,
        "active_connections": len(active_connections),
    }


# ── WebSocket endpoint ──────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)

    # Each client gets its own detector instance (independent signal buffer)
    detector = RPPGDetector(buffer_size=BUFFER_SIZE)
    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"bpm": 0, "spo2": 0, "status": "Bad JSON"})
                continue

            image_data = message.get("image")
            if not image_data:
                await websocket.send_json({"bpm": 0, "spo2": 0, "status": "No image field"})
                continue

            # Decode base64 → OpenCV frame (offloaded to thread pool)
            frame = await loop.run_in_executor(executor, base64_to_cv2, image_data)

            if frame is None:
                await websocket.send_json({"bpm": 0, "spo2": 0, "status": "Frame decode failed"})
                continue

            # Run detector (CPU-bound) off the event loop
            result = await loop.run_in_executor(executor, detector.process_frame, frame)
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )