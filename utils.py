import base64
import numpy as np
import cv2
import time

def base64_to_cv2(base64_string: str) -> np.ndarray:
    """
    Base64 string (image data) ko OpenCV BGR format mein badalne ke liye.
    """
    try:
        # Agar prefix (data:image/...) hai toh usse hatao
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Warning: imdecode return failed")
        return frame

    except Exception as e:
        print(f"Conversion error: {e}")
        return None

def normalize_signal(signal_array) -> np.ndarray:
    """
    Signal ko 0-1 ki range mein normalize karne ke liye.
    """
    arr = np.array(signal_array, dtype=float)
    if arr.size < 2: return arr
    
    lo, hi = arr.min(), arr.max()
    if hi == lo: return np.zeros_like(arr)
    
    return (arr - lo) / (hi - lo)

def get_roi_average(frame, x, y, w=20, h=20, channel="green"):
    """
    Kisi specific area (ROI) ka mean pixel value nikalna.
    """
    H, W = frame.shape[:2]
    # Boundary checks
    x1, x2 = max(0, x), min(W, x + w)
    y1, y2 = max(0, y), min(H, y + h)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return 0.0
    
    # BGR order: Blue=0, Green=1, Red=2
    idx = {"blue": 0, "green": 1, "red": 2}.get(channel.lower(), 1)
    return float(np.mean(roi[:, :, idx]))

def validate_frame(frame, min_size=(160, 160)):
    """
    Check karna ki frame valid hai aur resolution sahi hai ya nahi.
    """
    if frame is None or not isinstance(frame, np.ndarray):
        return False, "Invalid image object"
    
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False, "Not a BGR image"
        
    h, w = frame.shape[:2]
    if w < min_size[0] or h < min_size[1]:
        return False, f"Resolution too low: {w}x{h}"
        
    return True, "Valid"

def fps_calculator():
    """
    Real-time FPS calculate karne ke liye generator.
    """
    last = time.time()
    while True:
        now = time.time()
        dt = now - last
        fps = 1.0 / dt if dt > 0 else 0.0
        last = now
        yield fps