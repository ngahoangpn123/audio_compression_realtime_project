"""
Real-time Audio Streaming Client
Reads audio in chunks, streams PCM bytes qua WebSocket đến server,
và in metrics theo thời gian thực ra terminal.

Usage:
    python client/client.py --file data/sample.wav
    python client/client.py --generate --duration 15
"""

import argparse
import asyncio
import json
import sys
import os
import numpy as np
import websockets

import soundfile as sf   
import librosa           

# ── Terminal colours ──────────────────────────────────────────────────────────
RESET   = "\033[0m";  BOLD = "\033[1m";  DIM  = "\033[2m"
CYAN    = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
MAGENTA = "\033[95m"; RED  = "\033[91m"

SAMPLE_RATE   = 44100
CHUNK_SAMPLES = 4096
SERVER_URI    = "ws://localhost:8765"


# ─────────────────────────────────────────────────────────────────────────────
# Audio source helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_chunks(path: str, chunk_samples: int = CHUNK_SAMPLES):
    """
    Đọc file âm thanh và yield float32 PCM chunks.

    """
    try:
        info = sf.info(path)
        print(f"{DIM}Audio: {info.channels}ch · {info.samplerate} Hz · "
              f"{info.duration:.1f}s · {info.format}{RESET}")
    except Exception:
        pass

    samples, _sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)

    n_chunks = max(1, len(samples) // chunk_samples)
    for chunk in np.array_split(samples, n_chunks):
        if len(chunk) > 0:
            yield chunk.astype(np.float32)


def generate_sine_chunks(
    frequency: float = 440.0,
    duration_s: float = 10.0,
    chunk_samples: int = CHUNK_SAMPLES,
):
    """
    Tạo tín hiệu sine tổng hợp và yield từng chunk.
    """
    total_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, total_samples, endpoint=False, dtype=np.float32)
    signal = (
        0.50 * np.sin(2 * np.pi * frequency * t)
        + 0.25 * np.sin(2 * np.pi * frequency * 2 * t)
        + 0.15 * np.sin(2 * np.pi * frequency * 3 * t)
    )
    n_chunks = max(1, total_samples // chunk_samples)
    yield from (c for c in np.array_split(signal, n_chunks) if len(c) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_header():
    print(f"\n{BOLD}{CYAN}{'─'*70}{RESET}")
    print(f"{BOLD}{CYAN}  Audio Compression Real-Time Client{RESET}")
    print(f"{BOLD}{CYAN}{'─'*70}{RESET}")


def print_metrics(payload: dict, chunk_idx: int):
    ts        = payload.get("timestamp", 0)
    total_lat = payload.get("total_latency_ms", 0)
    codecs    = payload.get("codecs", {})

    print(f"\n{BOLD}Chunk #{chunk_idx:03d}{RESET}  "
          f"{DIM}t={ts:.3f}  total_latency={total_lat} ms{RESET}")

    for codec, m in codecs.items():
        if "error" in m:
            print(f"  {codec.upper():6s}  {RED}ERROR: {m['error']}{RESET}")
            continue
        snr     = m.get("snr_db", 0)
        psnr    = m.get("psnr_db", 0)
        ratio   = m.get("compression_ratio", 1)
        enc_lat = m.get("encode_latency_ms", 0)
        dec_lat = m.get("decode_latency_ms", 0)
        orig_kb = m.get("original_bytes", 0) / 1024
        comp_kb = m.get("compressed_bytes", 0) / 1024

        snr_col = GREEN if isinstance(snr, (int, float)) and snr > 25 else YELLOW
        print(
            f"  {BOLD}{codec.upper():6s}{RESET} "
            f"SNR={snr_col}{snr:>7.2f}dB{RESET}  "
            f"PSNR={psnr:>7.2f}dB  "
            f"Ratio={MAGENTA}{ratio:>5.2f}x{RESET}  "
            f"Enc={enc_lat:>6.1f}ms  Dec={dec_lat:>6.1f}ms  "
            f"{DIM}{orig_kb:.1f}KB->{comp_kb:.1f}KB{RESET}"
        )


def print_summary(history: list):
    """Thống kê tổng hợp"""
    if not history:
        return

    print(f"\n{BOLD}{CYAN}{'─'*70}{RESET}")
    print(f"{BOLD}Summary ({len(history)} chunks){RESET}")

    codec_data: dict = {}
    for payload in history:
        for codec, m in payload.get("codecs", {}).items():
            if "error" in m:
                continue
            d = codec_data.setdefault(
                codec, {"snr": [], "ratio": [], "enc_lat": [], "dec_lat": []}
            )
            d["snr"].append(m.get("snr_db", 0))
            d["ratio"].append(m.get("compression_ratio", 1))
            d["enc_lat"].append(m.get("encode_latency_ms", 0))
            d["dec_lat"].append(m.get("decode_latency_ms", 0))

    for codec, d in codec_data.items():
        print(
            f"  {BOLD}{codec.upper():6s}{RESET} "
            f"avgSNR={np.mean(d['snr']):>7.2f}dB  "
            f"avgRatio={np.mean(d['ratio']):>5.2f}x  "
            f"avgEnc={np.mean(d['enc_lat']):>6.1f}ms  "
            f"avgDec={np.mean(d['dec_lat']):>6.1f}ms"
        )
    print(f"{CYAN}{'─'*70}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main WebSocket coroutine
# ─────────────────────────────────────────────────────────────────────────────

async def stream_audio(uri: str, chunks, chunk_delay_s: float = 0.09,
                       save_results: str = None) -> list:
    """Kết nối server, stream audio chunks và thu thập metrics."""
    history = []
    try:
        async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
            print(f"{GREEN}Connected to {uri}{RESET}")
            await ws.send(json.dumps({"action": "ping"}))
            pong = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"{DIM}Server: {pong}{RESET}")

            for chunk_idx, chunk in enumerate(chunks):
                await ws.send(chunk.tobytes())
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=10)
                    payload = json.loads(resp)
                    if payload.get("type") == "metrics":
                        print_metrics(payload, chunk_idx)
                        history.append(payload)
                        
                        # --- GHI FILE SAU MỖI CHUNK ĐỂ DASHBOARD ĐỌC REAL-TIME ---
                        if save_results:
                            os.makedirs(os.path.dirname(save_results) or ".", exist_ok=True)
                            with open(save_results, "w") as f:
                                json.dump(history, f, indent=2)
                        # ---------------------------------------------------------

                except asyncio.TimeoutError:
                    print(f"{RED}Timeout chunk #{chunk_idx}{RESET}")
                
                await asyncio.sleep(chunk_delay_s)

    except ConnectionRefusedError:
        print(f"{RED}Cannot connect to {uri}. Is the server running?{RESET}")
        print(f"  Start with: {BOLD}python -m backend.server{RESET}")
        sys.exit(1)

    print_summary(history)

    if save_results and history:
        print(f"{GREEN}Results saved -> {save_results}{RESET}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real-time audio compression client")
    parser.add_argument("--file",       "-f", default="data/sample.wav")
    parser.add_argument("--generate",   "-g", action="store_true")
    parser.add_argument("--freq",             type=float, default=440.0)
    parser.add_argument("--duration",   "-d", type=float, default=10.0)
    parser.add_argument("--server",           default=SERVER_URI)
    parser.add_argument("--save",             default="results/client_metrics.json")
    parser.add_argument("--chunk-size",       type=int, default=CHUNK_SAMPLES)
    args = parser.parse_args()

    print_header()

    if args.generate:
        print(f"Generating {args.duration}s sine @ {args.freq} Hz...")
        chunks = generate_sine_chunks(args.freq, args.duration, args.chunk_size)
    else:
        if not os.path.exists(args.file):
            print(f"{RED}File not found: {args.file}{RESET}")
            sys.exit(1)
        print(f"Loading {args.file}...")
        chunks = load_audio_chunks(args.file, args.chunk_size)

    asyncio.run(stream_audio(args.server, chunks, save_results=args.save))


if __name__ == "__main__":
    main()
