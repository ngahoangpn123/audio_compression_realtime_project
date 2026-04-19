# 🎙 Real-Time Audio Compression Monitoring System

### 📚 Course: Multimedia Data Compression & Coding

**Project ID:** 25219 | **Class:** 168224
**Team Members:** Bui Quynh Hoa - 202414627 | Hoang Thi Phuong Nga - 202414648
**Instructor:** Dr. rer. nat. Pham Van Tien

---

> 📡 A real-time system for monitoring audio compression performance with an interactive dashboard.
> Supports comparison of multiple codecs (**MP3, AAC, Opus, OGG**) across **bitrate, SNR, PSNR, compression ratio, and latency**.

---

## 🚀 Overview

In modern multimedia systems, audio compression plays a critical role in reducing bandwidth while preserving perceptual quality. However, different codecs exhibit distinct trade-offs between **compression efficiency, signal fidelity, and latency**.

This project implements a **real-time audio compression monitoring system** that:

* Streams audio using WebSocket
* Applies multiple codecs via **FFmpeg (real implementations)**
* Computes metrics per chunk
* Visualizes results in an interactive dashboard

> ⚠️ This system uses **true codec execution (FFmpeg)** instead of simulated compression, ensuring realistic evaluation.

---

## ✨ Key Features

* 🎧 Real-time PCM audio streaming
* ⚙️ True codec processing (MP3, AAC, Opus, OGG)
* 📊 Live metrics:

  * SNR, PSNR
  * Compression Ratio
  * Latency
  * Bitrate
* 📈 Interactive dashboard (Plotly Dash)
* 🎼 Spectrogram & waveform visualization
* 🔊 Audio playback comparison
* 💾 JSON export for reproducibility

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

### 🔹 Components

* **Client:** Streams PCM float32 audio chunks
* **WebSocket Server:** Handles real-time communication
* **Codec Module (`audio_codec.py`):** FFmpeg-based encoding/decoding
* **Metrics Module (`metrics.py`):** Computes SNR, PSNR, LSD, compression ratio, latency
* **Dashboard:** Interactive visualization
* **Terminal:** JSON metric logs

---

## 🔄 Processing Pipeline

1. Audio input is loaded using `librosa` and converted to mono (44.1 kHz)
2. Signal is split into PCM float32 chunks (default: 4096 samples)
3. Each chunk is:

   * Encoded using selected codec
   * Decoded back into reconstructed audio
4. Original vs reconstructed → metric computation
5. Results streamed to dashboard and terminal in real time

---

## 🧠 Methodology

### Algorithmic Processing Pipeline

To ensure realistic evaluation, the system avoids naive array manipulation and leverages **FFmpeg via `pydub`**.

Each audio chunk undergoes:

1. **Normalization**
   Clip values to `[-1.0, 1.0]` to avoid distortion

2. **Encoding**
   Convert to compressed format using in-memory buffers (`io.BytesIO`)

3. **Decoding**
   Decode compressed bitstream back to PCM

4. **Metric Computation**

   * SNR (Signal-to-Noise Ratio)
   * PSNR (Peak SNR)
   * LSD (Log-Spectral Distortion)
   * Compression Ratio
   * Latency

> ✅ Ensures accurate evaluation of real codec behavior.

---

## ⚙️ Setup

### 1. Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg -y

# macOS
brew install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH
```

---

### 2. Python Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

### 3. Generate Sample Audio (Optional)

```bash
python data/generate_sample.py
python data/download_dataset.py
```

---

## 🚀 Running the System

### 🔥 Option A — Full Real-Time System (Recommended)

**Terminal 1 — Start Server**

```bash
python -m backend.server
```

**Terminal 2 — Start Dashboard**

```bash
python frontend/app.py
```

**Terminal 3 — Run Client**

```bash
python client/client.py --file data/sample.wav
python client/client.py --file data/speech_test.wav
python client/client.py --file data/music_test.wav
python client/client.py --file data/percussion_test.wav
python client/client.py --file data/noise_test.wav
```

---

### 📊 Option B — Dashboard Only (Offline Analysis)

```bash
python frontend/app.py
```

---

## 🎛 Client Options

```bash
python client/client.py [OPTIONS]

--file FILE         Path to WAV audio file
--generate          Use synthetic audio
--freq FLOAT        Frequency (default: 440 Hz)
--duration FLOAT    Duration in seconds
--server URI        WebSocket server address
--save PATH         Save JSON results
--chunk-size INT    Samples per chunk (default: 4096)
```

---

## 📊 Metrics

| Metric            | Description                 |
| ----------------- | --------------------------- |
| SNR (dB)          | Signal quality vs noise     |
| PSNR (dB)         | Peak distortion             |
| LSD (dB)          | Frequency-domain distortion |
| Compression Ratio | Size reduction              |
| Space Saving (%)  | Storage efficiency          |
| Latency (ms)      | Encode + decode time        |

---

## 📈 Expected Results

| Codec | SNR (dB) | Compression Ratio | Latency  |
| ----- | -------- | ----------------- | -------- |
| MP3   | ~33.26   | ~11.0×            | ~5.2 ms  |
| AAC   | ~-3.2*   | ~11.8×            | ~12.1 ms |
| Opus  | ~34.81   | ~11.5×            | ~10.5 ms |
| OGG   | ~34.12   | ~11.2×            | ~6.8 ms  |

> *AAC may show negative SNR due to phase shift effects, but perceptual quality remains high.*

---

## 🗂 Project Structure

```
audio_compression_realtime_project/
│
├── backend/
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

## 🔬 Reproducibility

### Environment Setup
```bash
sudo apt install ffmpeg -y
python -m venv .venv && source .venv/bin/activate pip install -r requirements.txt
```

---
### Dataset preparation
```bash
python data/generate_sample.py
python data/download_dataset.py
``` 

---
### Real-time mode (three separate terminals)
```bash
python -m backend.server # Terminal 1: WebSocket server
python frontend/app.py # Terminal 2: Dashboard → http://localhost:8050 python
client/client.py --generate # Terminal 3: Stream synthetic audio
```

---
### Batch evaluation
```bash
python evaluate.py --file data/sample.wav
python evaluate.py --file data/speech_test.wav
python evaluate.py --file data/music_test.wav
python evaluate.py --file data/percussion_test.wav
python evaluate.py --file data/noise_test.wav
# Outputs: results/evaluation_results.json, results/*.png
```

---

## 👥 Team Contribution

| Member | Contribution |
|--------|-------------|
| **Bui Quynh Hoa** | • Designed and implemented core backend system (`server.py`, `audio_codec.py`) <br> • Developed data generation & dataset scripts (`generate_sample.py`, `download_dataset.py`) <br> • Led quantitative evaluation (Latency, SNR, Compression Ratio) via automated testing <br> • Authored technical documentation and `README.md` |
| **Hoang Thi Phuong Nga** | • Developed interactive dashboard (`app.py`) with real-time spectrogram & waveform visualization <br> • Implemented metrics module (`metrics.py`) and evaluation pipeline (`evaluate.py`) <br> • Led visual & comparative analysis across codecs <br> • Developed client streaming logic (`client.py`) and managed dependencies (`requirements.txt`) |
| **Both Authors** | • Co-designed system architecture and processing pipeline <br> • Integrated all modules into a full end-to-end system <br> • Designed experimental framework (bitrate sweep, codec comparison) <br> • Performed testing, debugging, and optimization <br> • Jointly analyzed results and finalized report |

---


