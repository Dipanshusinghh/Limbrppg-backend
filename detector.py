"""
LimbVital rPPG Engine - Heart Rate Extraction
"""
import cv2
import numpy as np
import time
import os
import urllib.request
from scipy import signal

# Mediapipe setup (Latest Tasks API)
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def _download_model(path: str) -> bool:
    # Agar model file nahi hai toh download karlo
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        return True
    except Exception:
        return False

def _init_face_landmarker():
    if not MEDIAPIPE_AVAILABLE: return None
    if not os.path.exists(MODEL_PATH):
        if not _download_model(MODEL_PATH): return None
    
    try:
        base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5
        )
        return mp_vision.FaceLandmarker.create_from_options(options)
    except Exception:
        return None

def _init_opencv_fallback():
    # Agar mediapipe fail ho jaye toh purana Haar cascade use karo
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    return None if cascade.empty() else cascade

class RPPGDetector:
    # Landmarks: forehead(10), left_cheek(234), right_cheek(454)
    ROI_LANDMARKS = [10, 234, 454]

    def __init__(self, buffer_size: int = 150):
        self.buffer_size = buffer_size
        self.green_values, self.times = [], []
        
        # Detector choose karo
        self.landmarker = _init_face_landmarker()
        self.use_mediapipe = self.landmarker is not None
        self.cascade = None if self.use_mediapipe else _init_opencv_fallback()

    def _get_green_mediapipe(self, frame, landmarks):
        h, w = frame.shape[:2]
        means = []
        for idx in self.ROI_LANDMARKS:
            lm = landmarks[idx]
            cx, cy, r = int(lm.x * w), int(lm.y * h), 15
            roi = frame[max(0, cy-r):min(h, cy+r), max(0, cx-r):min(w, cx+r)]
            if roi.size > 0:
                means.append(np.mean(roi[:, :, 1])) # Green channel extraction
        return float(np.mean(means)) if means else None

    def _get_green_opencv(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0: return None
        
        x, y, w, h = faces[0]
        # Forehead region approx logic
        roi = frame[y+int(h*0.05):y+int(h*0.25), x+w//3:x+2*w//3]
        return float(np.mean(roi[:, :, 1])) if roi.size > 0 else None

    def process_frame(self, frame: np.ndarray):
        if frame is None: return {"bpm": 0, "status": "No Frame"}
        
        if self.use_mediapipe:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.landmarker.detect(mp_img)
            if not res.face_landmarks: 
                self._reset_buffer()
                return {"bpm": 0, "status": "Searching..."}
            green = self._get_green_mediapipe(frame, res.face_landmarks[0])
        else:
            green = self._get_green_opencv(frame)
            
        return self._update_buffer(green)

    def _update_buffer(self, green):
        if green is None:
            self._reset_buffer()
            return {"bpm": 0, "status": "Face not clear"}
            
        self.green_values.append(green)
        self.times.append(time.time())

        if len(self.green_values) > self.buffer_size:
            self.green_values.pop(0)
            self.times.pop(0)

        if len(self.green_values) == self.buffer_size:
            return self._calculate_vitals()
        
        return {"bpm": 0, "status": f"Calibrating {int(len(self.green_values)/self.buffer_size*100)}%"}

    def _reset_buffer(self):
        self.green_values.clear()
        self.times.clear()

    def _calculate_vitals(self):
        try:
            fps = (len(self.times)-1) / (self.times[-1] - self.times[0])
            nyq = 0.5 * fps
            # Filtering band: 45 to 210 BPM
            low, high = 0.75/nyq, 3.5/nyq
            
            if not (0 < low < high < 1): return {"bpm": 0, "status": "FPS issue"}

            detrended = signal.detrend(self.green_values)
            b, a = signal.butter(3, [low, high], btype="bandpass")
            filtered = signal.filtfilt(b, a, detrended)
            
            # FFT se frequency nikalna
            fft_vals = np.abs(np.fft.rfft(filtered))
            freqs = np.fft.rfftfreq(len(filtered), 1.0/fps)
            
            valid = (freqs >= 0.75) & (freqs <= 3.5)
            bpm = int(round(freqs[valid][np.argmax(fft_vals[valid])] * 60))
            
            return {
                "bpm": max(45, min(210, bpm)),
                "stress": self._calculate_stress(filtered, fps),
                "status": "Stable" if 60 <= bpm <= 100 else "Check Status"
            }
        except Exception:
            return {"bpm": 0, "status": "Calc error"}

    def _calculate_stress(self, sig, fps):
        # Heart rate variability se stress estimate karna
        try:
            peaks, _ = signal.find_peaks(sig, distance=int(fps*0.4))
            if len(peaks) < 4: return 0
            sdnn = np.std(np.diff(peaks) / fps * 1000)
            return int(np.clip(100 - (sdnn - 20) * 1.1, 10, 90))
        except: return 0