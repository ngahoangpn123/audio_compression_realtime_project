# 🎙 Monitoring Audio Compression in Real Time with Interactive Dashboards

> **Học phần:** Nén và Mã hóa Dữ liệu Đa phương tiện  
> **Mã project:** 25219  
> **Mô tả:** Hệ thống giám sát nén âm thanh theo thời gian thực với dashboard tương tác, hỗ trợ so sánh các codec (MP3, AAC, Opus, OGG) theo bitrate với các chỉ số SNR, PSNR, tỉ lệ nén, và latency.

---

## 📐 System Architecture

```
┌─────────────┐   PCM chunks   ┌──────────────────┐   metrics JSON
│  Client      │ ────────────▶ │  WebSocket Server │ ──────────────▶ Terminal
│ (client.py)  │               │  (server.py)      │
└─────────────┘               │  audio_codec.py   │
                               │  metrics.py       │
                               └──────────────────┘
                                        │
                               ┌──────────────────┐
                               │  Dash Dashboard  │
                               │  (frontend/app)  │
                               └──────────────────┘
```

**Pipeline:**
1. Audio input → chunked PCM float32
2. Each chunk → encode with codec (MP3/AAC/Opus/OGG) → decode → reconstruct
3. Compute SNR, PSNR, compression ratio, latency per chunk
4. Metrics streamed to dashboard & terminal in real time
5. Dashboard plots bitrate sweeps, spectrograms, waveform comparison, playback

---

## ⚙️ Setup

### 1. System dependency — ffmpeg

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg -y

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

### 2. Python environment

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate sample audio (optional)

```bash
python data/generate_sample.py
# Creates data/sample.wav  (10s synthetic speech-like signal)
```

---

## 🚀 Running the System

### Option A — Full Real-Time System (recommended)

**Terminal 1: Start the WebSocket server**
```bash
python -m backend.server
# Server listens on ws://0.0.0.0:8765
```

**Terminal 2: Start the dashboard**
```bash
python frontend/app.py
# Dashboard at http://localhost:8050
```

**Terminal 3: Stream audio from client**
```bash
# Stream a WAV file
python client/client.py --file data/sample.wav

# Or use synthetic audio (no WAV file needed)
python client/client.py --generate --duration 15
```

### Option B — Dashboard only (offline analysis)

```bash
python frontend/app.py
# Open http://localhost:8050
# Upload a WAV file and click "▶ Analyze"
```

---

## 🎛 Client Options

```
python client/client.py [OPTIONS]

  --file FILE        Path to WAV audio file (default: data/sample.wav)
  --generate         Use synthetic sine-wave audio (no file needed)
  --freq FLOAT       Sine frequency in Hz (default: 440)
  --duration FLOAT   Duration in seconds for synthetic audio (default: 10)
  --server URI       WebSocket server address (default: ws://localhost:8765)
  --save PATH        Save metrics JSON to path (default: results/client_metrics.json)
  --chunk-size INT   PCM samples per chunk (default: 4096 ≈ 93ms at 44100 Hz)
```

---

## 📊 Metrics

| Metric | Description |
|--------|-------------|
| **SNR (dB)** | Signal-to-Noise Ratio — higher is better |
| **PSNR (dB)** | Peak SNR — measures max distortion |
| **Compression Ratio** | Original size / Compressed size |
| **Space Saving (%)** | `(1 - comp/orig) × 100` |
| **Encode Latency (ms)** | Time to compress one chunk |
| **Decode Latency (ms)** | Time to decompress one chunk |
| **Effective Bitrate (kbps)** | Actual bits per second achieved |

---

## 🗂 Project Structure

```
audio-compression-realtime/
│
├── backend/
│   ├── __init__.py         # Package init
│   ├── server.py           # Async WebSocket server (asyncio + websockets)
│   ├── audio_codec.py      # MP3/AAC/Opus/OGG encode-decode wrappers (pydub/ffmpeg)
│   └── metrics.py          # SNR, PSNR, spectrogram, compression metrics
│
├── frontend/
│   └── app.py              # Plotly Dash interactive dashboard
│
├── client/
│   └── client.py           # Streaming client — reads WAV, sends PCM, prints metrics
│
├── data/
│   ├── generate_sample.py  # Generates data/sample.wav
│   └── sample.wav          # (generated) Test audio
│
├── results/                # Auto-created; stores metric JSONs from client runs
│
├── docs/
│   ├── report.pdf          # Technical report
│   └── slides.pdf          # Presentation slides
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔬 Reproducing Results

```bash
# 1. Generate test audio
python data/generate_sample.py

# 2. Start server
python -m backend.server &

# 3. Run client — saves metrics to results/
python client/client.py --file data/sample.wav --save results/run1.json

# 4. Inspect results
python -c "
import json
data = json.load(open('results/run1.json'))
print(f'{len(data)} chunks processed')
for codec, m in data[0]['codecs'].items():
    print(f'{codec}: SNR={m[\"snr_db\"]}dB, ratio={m[\"compression_ratio\"]}x')
"
```

---

## 📈 Expected Results

| Codec | SNR @ 128 kbps | Compression Ratio | Encode Latency |
|-------|---------------|-------------------|----------------|
| MP3   | ~28–35 dB     | 8–12×             | 5–15 ms        |
| AAC   | ~30–38 dB     | 9–14×             | 8–20 ms        |
| Opus  | ~32–40 dB     | 10–16×            | 3–10 ms        |
| OGG   | ~28–36 dB     | 8–12×             | 6–18 ms        |

*Values vary with input audio type (speech vs music) and hardware.*

---

## 🧩 Key Implementation Details

### Codec Pipeline (`backend/audio_codec.py`)
- **Input:** `float32` numpy array, values in `[-1, 1]`
- **Encode:** `pydub.AudioSegment.export()` → ffmpeg subprocess → compressed bytes
- **Decode:** `pydub.AudioSegment.from_file()` → ffmpeg → `int16` → `float32`
- **Fallback:** If pydub/ffmpeg unavailable, returns lossless WAV (for metric testing)

### WebSocket Protocol (`backend/server.py`)
- **Binary frame** → PCM float32 bytes → triggers compression + metric response
- **Text frame (JSON)** → control command (`ping`, `get_codecs`, `compress_file`)
- **Response JSON:**
  ```json
  {
    "type": "metrics",
    "timestamp": 1700000000.0,
    "chunk_samples": 4096,
    "total_latency_ms": 45.2,
    "codecs": {
      "mp3": { "snr_db": 32.1, "compression_ratio": 10.2, ... }
    }
  }
  ```

### Dashboard (`frontend/app.py`)
- **Upload** → decoded to float32 → stored in `dcc.Store`
- **Analyze** → bitrate sweep (32–320 kbps) × all selected codecs
- **Charts:** SNR vs bitrate (line), compression ratio (bar), latency (bar), waveform, spectrograms
- **Playback:** `html.Audio` with `data:audio/wav;base64,...` src

---

## 👥 Team Contribution

| Thành viên | Đóng góp |
|------------|----------|
| *(điền tên)* | Backend server, codec module |
| *(điền tên)* | Frontend dashboard, visualization |
| *(điền tên)* | Client, metrics, evaluation |
| *(điền tên)* | Report, slides, testing |

---

## 📄 License

MIT License — for educational use.
