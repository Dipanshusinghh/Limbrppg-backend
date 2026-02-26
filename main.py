import os
import json
import asyncio
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Local files se import
from detector import RPPGDetector
from utils import base64_to_cv2

load_dotenv()

# Env variables setup
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", 150))

app = FastAPI(title="LimbVital API")

# CORS allow karna zaroori hai taaki frontend se connection block na ho
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# CPU-intensive tasks ke liye thread pool (Event loop free rakhne ke liye)
executor = ThreadPoolExecutor(max_workers=4)
active_connections = set()

@app.get("/")
async def status():
    return {"service": "LimbVital Engine", "active": True}

@app.get("/health")
async def health_check():
    # Basic health check for monitoring
    return {
        "status": "ok",
        "connections": len(active_connections)
    }

# --- WebSocket Logic ---

@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    
    # Har client ka apna buffer maintain hoga
    detector = RPPGDetector(buffer_size=BUFFER_SIZE)
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                payload = json.loads(data)
            except:
                await websocket.send_json({"status": "Invalid JSON"})
                continue

            img_b64 = payload.get("image")
            if not img_b64:
                continue

            # Base64 ko OpenCV frame mein convert karo (Thread pool mein)
            frame = await loop.run_in_executor(executor, base64_to_cv2, img_b64)

            if frame is not None:
                # Frame process karke result nikalna
                result = await loop.run_in_executor(executor, detector.process_frame, frame)
                await websocket.send_json(result)

    except WebSocketDisconnect:
        print("Client disconnected normally")
    except Exception as e:
        print(f"Server Error: {e}")
    finally:
        active_connections.discard(websocket)

if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=DEBUG)