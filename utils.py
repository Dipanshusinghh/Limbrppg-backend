"""
LimbVital rPPG Engine – Utility Functions
"""

import base64
import numpy as np
import cv2


def base64_to_cv2(base64_string: str) -> np.ndarray | None:
    """
    Convert a Base64-encoded image string (with or without data-URL prefix)
    to an OpenCV BGR image.

    Args:
        base64_string: e.g. "data:image/jpeg;base64,/9j/4AAQ..." or raw base64

    Returns:
        BGR ndarray, or None if decoding fails.
    """
    try:
        # Strip optional data-URL prefix
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("⚠️  cv2.imdecode returned None – check image data")

        return frame

    except base64.binascii.Error as e:
        print(f"❌ Base64 decode error: {e}")
        return None
    except Exception as e:
        print(f"❌ Image conversion error: {e}")
        return None


def normalize_signal(signal_array) -> np.ndarray:
    """
    Normalize a signal to the [0, 1] range.

    Args:
        signal_array: list or ndarray of floats

    Returns:
        Normalized ndarray.
    """
    arr = np.array(signal_array, dtype=float)
    if arr.size < 2:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def get_roi_average(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int = 20,
    h: int = 20,
    channel: str = "green",
) -> float:
    """
    Return the mean pixel value of a colour channel inside a ROI.

    Args:
        frame:   BGR image
        x, y:   top-left corner of ROI
        w, h:   ROI dimensions
        channel: 'red', 'green', or 'blue'

    Returns:
        Mean channel value (0–255).
    """
    H, W = frame.shape[:2]
    x1, x2 = max(0, x), min(W, x + w)
    y1, y2 = max(0, y), min(H, y + h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    channel_idx = {"blue": 0, "green": 1, "red": 2}.get(channel.lower(), 1)
    return float(np.mean(roi[:, :, channel_idx]))


def validate_frame(
    frame, min_size: tuple[int, int] = (160, 160)
) -> tuple[bool, str]:
    """
    Check that a frame is a valid BGR image with a minimum resolution.

    Returns:
        (True, "Valid") or (False, <reason>)
    """
    if frame is None:
        return False, "Frame is None"
    if not isinstance(frame, np.ndarray):
        return False, "Frame is not a numpy array"
    if frame.ndim != 3 or frame.shape[2] != 3:
        return False, "Frame must be a 3-channel (BGR) image"
    h, w = frame.shape[:2]
    if w < min_size[0] or h < min_size[1]:
        return False, f"Frame too small: {w}×{h} (min {min_size[0]}×{min_size[1]})"
    return True, "Valid"


def fps_calculator():
    """
    Generator that yields instantaneous FPS each time next() is called.

    Usage:
        calc = fps_calculator()
        while True:
            fps = next(calc)
    """
    import time
    last = time.time()
    while True:
        now = time.time()
        dt = now - last
        fps = 1.0 / dt if dt > 0 else 0.0
        last = now
        yield fps