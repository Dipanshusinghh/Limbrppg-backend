"""
LimbVital rPPG Engine - Face Detection & Heart Rate Extraction
Compatible with mediapipe >= 0.10.x (Tasks API)
Falls back to OpenCV Haar cascade if mediapipe model is unavailable.
"""
import cv2
import numpy as np
import time
import os
import urllib.request
from scipy import signal
# ── Mediapipe Tasks API (0.10.x+) ──────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
# Path where the model file will be stored (next to this script)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
def _download_model(path: str) -> bool:
    """Download the face_landmarker.task model if not present."""
    print(f"📥 Downloading face landmarker model → {path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        print("✅ Model downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Model download failed: {e}")
        return False
def _init_face_landmarker():
    """
    Initialise MediaPipe FaceLandmarker (Tasks API).
    Returns a landmarker object or None on failure.
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH):
        ok = _download_model(MODEL_PATH)
        if not ok:
            return None
    try:
        base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print("✅ LimbVital Engine: FaceLandmarker (Tasks API) initialised")
        return landmarker
    except Exception as e:
        print(f"❌ FaceLandmarker init failed: {e}")
        return None
def _init_opencv_fallback():
    """Load OpenCV Haar cascade as fallback face detector."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("❌ OpenCV Haar cascade could not be loaded")
        return None
    print("⚠️  LimbVital Engine: Using OpenCV fallback (no MediaPipe model)")
    return cascade
# ─────────────────────────────────────────────────────────────────────────────
class RPPGDetector:
    """
    Remote Photoplethysmography detector.
    Landmarks used for ROI sampling (forehead, left cheek, right cheek):
        10  – forehead centre
        234 – left cheek
        454 – right cheek
    """
    ROI_LANDMARKS = [10, 234, 454]
    def __init__(self, buffer_size: int = 150):
        self.buffer_size = buffer_size
        self.green_values: list[float] = []
        self.times: list[float] = []
        # Try MediaPipe first, fall back to OpenCV
        self.landmarker = _init_face_landmarker()
        self.use_mediapipe = self.landmarker is not None
        if not self.use_mediapipe:
            self.cascade = _init_opencv_fallback()
        else:
            self.cascade = None
    # ── Internal helpers ──────────────────────────────────────────────────
    def _get_green_mediapipe(self, frame: np.ndarray, landmarks) -> float | None:
        """Extract mean green value from forehead + cheek ROIs using landmarks."""
        h, w = frame.shape[:2]
        means = []
        for idx in self.ROI_LANDMARKS:
            lm = landmarks[idx]
            cx, cy = int(lm.x * w), int(lm.y * h)
            r = 15
            roi = frame[max(0, cy - r): min(h, cy + r),
                        max(0, cx - r): min(w, cx + r)]
            if roi.size > 0:
                means.append(float(np.mean(roi[:, :, 1])))  # green channel
        return float(np.mean(means)) if means else None
    def _get_green_opencv(self, frame: np.ndarray) -> float | None:
        """
        Fallback: detect face with Haar cascade, sample forehead ROI.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        # Forehead ≈ top 20 % of face bounding box, centre third horizontally
        fx1 = x + w // 3
        fx2 = x + 2 * w // 3
        fy1 = y + int(h * 0.05)
        fy2 = y + int(h * 0.25)
        roi = frame[fy1:fy2, fx1:fx2]
        if roi.size == 0:
            return None
        return float(np.mean(roi[:, :, 1]))
    # ── Public API ────────────────────────────────────────────────────────
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Accepts a BGR OpenCV frame and returns a vitals dict:
            { "bpm": int, "spo2": int, "status": str }
        """
        if frame is None:
            return {"bpm": 0, "spo2": 0, "status": "Invalid frame"}
        if self.use_mediapipe:
            return self._process_mediapipe(frame)
        elif self.cascade is not None:
            return self._process_opencv(frame)
        else:
            return {"bpm": 0, "spo2": 0, "status": "Engine Error: no detector"}
    def _process_mediapipe(self, frame: np.ndarray) -> dict:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)
            if not result.face_landmarks:
                self._reset_buffer()
                return {"bpm": 0, "spo2": 0, "status": "Searching for face…"}
            landmarks = result.face_landmarks[0]  # list of NormalizedLandmark
            green = self._get_green_mediapipe(frame, landmarks)
            return self._update_buffer(green)
        except Exception as e:
            return {"bpm": 0, "spo2": 0, "status": f"MediaPipe error: {e}"}
    def _process_opencv(self, frame: np.ndarray) -> dict:
        try:
            green = self._get_green_opencv(frame)
            return self._update_buffer(green)
        except Exception as e:
            return {"bpm": 0, "spo2": 0, "status": f"OpenCV error: {e}"}
    def _update_buffer(self, green: float | None) -> dict:
        if green is None:
            self._reset_buffer()
            return {"bpm": 0, "spo2": 0, "status": "No face / adjust lighting"}
        self.green_values.append(green)
        self.times.append(time.time())
        # Rolling window
        if len(self.green_values) > self.buffer_size:
            self.green_values.pop(0)
            self.times.pop(0)
        if len(self.green_values) == self.buffer_size:
            return self._calculate_vitals()
        progress = int(len(self.green_values) / self.buffer_size * 100)
        return {"bpm": 0, "spo2": 0, "status": f"Calibrating… {progress}%"}
    def _reset_buffer(self):
        self.green_values.clear()
        self.times.clear()
    def _calculate_vitals(self) -> dict:
        try:
            duration = self.times[-1] - self.times[0]
            if duration <= 0:
                return {"bpm": 0, "spo2": 0, "status": "Waiting for signal"}
            fps = (len(self.times) - 1) / duration
            nyquist = 0.5 * fps
            # Safety: bandpass limits must be within (0, 1) relative to Nyquist
            low = 0.75 / nyquist   # ≈ 45 BPM
            high = 3.5 / nyquist   # ≈ 210 BPM
            if low >= 1.0 or high >= 1.0 or low >= high:
                return {"bpm": 0, "spo2": 0, "status": "Low FPS – move closer"}
            detrended = signal.detrend(self.green_values)
            b, a = signal.butter(3, [low, high], btype="bandpass")
            filtered = signal.filtfilt(b, a, detrended)
            fft_vals = np.abs(np.fft.rfft(filtered))
            freqs = np.fft.rfftfreq(len(filtered), 1.0 / fps)
            # Restrict peak search to 45-210 BPM band
            valid = (freqs >= 0.75) & (freqs <= 3.5)
            if not np.any(valid):
                return {"bpm": 0, "spo2": 0, "status": "Signal too weak"}
            peak_freq = freqs[valid][np.argmax(fft_vals[valid])]
            bpm = int(round(peak_freq * 60))
            bpm = max(45, min(210, bpm))
            # HRV-based stress calculation
            stress = self._calculate_stress(filtered, fps)
            return {
                "bpm": bpm,
                "spo2": int(np.random.randint(97, 100)),  # Simulated placeholder
                "stress": stress,
                "status": self._get_status(bpm),
            }
        except Exception as e:
            return {"bpm": 0, "spo2": 0, "status": f"Calc error: {e}"}

    def _calculate_stress(self, filtered_signal: np.ndarray, fps: float) -> int:
        """
        HRV-based stress estimation (0-100).
        Uses SDNN — Standard Deviation of NN intervals.
        Low SDNN = high stress, High SDNN = low stress.
        """
        try:
            # Find peaks (heartbeats) in filtered signal
            min_distance = int(fps * 0.4)  # minimum 0.4s between beats (max 150 BPM)
            peaks, _ = signal.find_peaks(filtered_signal, distance=min_distance)
            if len(peaks) < 4:
                return 0  # Not enough peaks to calculate HRV
            # NN intervals = time between consecutive beats (in ms)
            nn_intervals = np.diff(peaks) / fps * 1000
            # SDNN — standard deviation of NN intervals
            sdnn = float(np.std(nn_intervals))
            # Convert SDNN to stress score (0-100)
            # SDNN > 100ms = very relaxed (stress ~10)
            # SDNN < 20ms  = very stressed (stress ~90)
            stress = int(np.clip(100 - (sdnn - 20) * (90 / 80), 10, 90))
            return stress
        except Exception:
            return 0
    def _get_status(self, bpm: int) -> str:
        """Return health status based on BPM."""
        if bpm < 50:
            return "⚠️ Very Low — Check Now"
        elif bpm < 60:
            return "🟡 Low Heart Rate"
        elif bpm <= 100:
            return "🟢 Stable"
        elif bpm <= 120:
            return "🟡 Elevated"
        else:
            return "🔴 High — Take Rest"