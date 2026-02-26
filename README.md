# 🫀 LimbVital rPPG Engine

Real-time heart rate detection using camera-based remote photoplethysmography (rPPG).

## 📋 Features

- ✅ Real-time heart rate (BPM) detection
- ✅ Signal quality monitoring
- ✅ SpO2 simulation (camera-based approximation)
- ✅ Stress level estimation (HRV-based)
- ✅ WebSocket-based communication
- ✅ Python 3.13 compatible
- ✅ Production-ready error handling

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python main.py
```

Server will start at:
- **HTTP**: `http://0.0.0.0:8000`
- **WebSocket**: `ws://0.0.0.0:8000/ws`
- **Health Check**: `http://0.0.0.0:8000/health`

## 📡 API Endpoints

### WebSocket: `/ws`

**Send (Frontend → Backend):**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Receive (Backend → Frontend):**
```json
{
  "bpm": 72,
  "signal": 0.65,
  "spo2": 98,
  "stress": 15,
  "status": "Healthy / Signal Stable"
}
```

### HTTP: `/health`

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "mediapipe": "operational",
  "active_connections": 1
}
```

## 🔧 Configuration

Edit `.env` file:

```env
HOST=0.0.0.0           # Server host
PORT=8000              # Server port
DEBUG=True             # Enable debug mode
BUFFER_SIZE=150        # Signal buffer size
MIN_BPM=45             # Minimum valid BPM
MAX_BPM=210            # Maximum valid BPM
```

## 📊 How It Works

### 1. Face Detection
Uses MediaPipe Face Mesh to detect forehead region (Landmark #10)

### 2. Signal Extraction
Extracts green channel values (hemoglobin absorption peak)

### 3. Signal Processing
- **Detrending**: Removes baseline drift
- **Bandpass Filter**: 0.7-3.5 Hz (42-210 BPM)
- **FFT**: Identifies dominant frequency

### 4. Vital Calculation
- **BPM**: Frequency × 60
- **HRV Stress**: Based on SDNN (Standard Deviation of NN intervals)
- **SpO2**: Simulated (real SpO2 needs dual-wavelength)

## 🧪 Testing

### Test WebSocket Connection

```javascript
// Browser Console
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  console.log('✅ Connected to LimbVital Engine');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('BPM:', data.bpm);
};

// Send test frame (replace with actual base64 image)
ws.send(JSON.stringify({
  image: "data:image/jpeg;base64,..."
}));
```

### Test with cURL

```bash
# Health check
curl http://localhost:8000/health

# Root endpoint
curl http://localhost:8000/
```

## 📁 Project Structure

```
limbvital-backend/
│
├── main.py              # FastAPI server & WebSocket handler
├── detector.py          # rPPG detection & signal processing
├── utils.py             # Helper functions (base64, normalization)
├── .env                 # Configuration file
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🐛 Troubleshooting

### Issue: "Mediapipe critical load error"
**Solution**: Install mediapipe
```bash
pip install mediapipe==0.10.14
```

### Issue: "Connection refused"
**Solution**: Check if server is running and HOST is `0.0.0.0`

### Issue: "Frame decode failed"
**Solution**: Ensure base64 image includes proper prefix:
```
data:image/jpeg;base64,<base64_string>
```

### Issue: "Signal quality poor"
**Checklist:**
- ✅ Good lighting (not too bright/dark)
- ✅ Face clearly visible
- ✅ Minimal movement
- ✅ Camera focused on face

## 🔒 Security Notes

### For Production:

1. **Change CORS Origins**:
```python
# In main.py
allow_origins=["https://yourdomain.com"]
```

2. **Add Rate Limiting**:
```bash
pip install slowapi
```

3. **Enable HTTPS**:
```bash
uvicorn main:app --host 0.0.0.0 --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

4. **Environment Variables**:
Never commit `.env` to version control. Add to `.gitignore`:
```
.env
__pycache__/
*.pyc
```

## 📝 License

MIT License - Free for personal and commercial use

## 🤝 Contributing

Issues and PRs welcome!

## 📧 Support

For questions: Open an issue on GitHub

---

**Made with ❤️ for LimbVital**
