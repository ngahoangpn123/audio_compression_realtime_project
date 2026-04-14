"""
Real-time Audio Streaming Client
Reads a WAV file in chunks, sends PCM bytes over WebSocket to the server,
and prints live compression metrics to the terminal.

Usage:
    python client/client.py --file data/sample.wav --codec mp3 --bitrate 128
    python client/client.py --generate  # use synthetic sine-wave audio
"""

import argparse
import asyncio
import json
import time
import wave
import sys
import io
import os
import numpy as np
import websockets

# ── Colours for terminal output ──────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
DIM = "\033[2m"

SAMPLE_RATE = 44100
CHUNK_SAMPLES = 4096
SERVER_URI = "ws://localhost:8765"


# ─────────────────────────────────────────────────────────────────────────────
# Audio source helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_wav_chunks(path: str, chunk_samples: int = CHUNK_SAMPLES):
    """Yield float32 PCM chunks from a WAV file."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        print(f"{DIM}WAV info: {n_channels}ch, {sampwidth*8}-bit, {framerate} Hz{RESET}")

        while True:
            raw = wf.readframes(chunk_samples)
            if not raw:
                break
            dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
            arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
            if sampwidth == 1:
                arr = (arr - 128) / 128.0
            else:
                arr /= 2 ** (sampwidth * 8 - 1)
            if n_channels > 1:
                arr = arr[::n_channels]  # take left channel
            yield arr.astype(np.float32)


def generate_sine_chunks(
    frequency: float = 440.0,
    duration_s: float = 10.0,
    chunk_samples: int = CHUNK_SAMPLES,
):
    """Yield synthetic sine-wave chunks (for testing without a WAV file)."""
    total_samples = int(SAMPLE_RATE * duration_s)
    t = np.arange(total_samples) / SAMPLE_RATE
    # Mix of sine waves for richer test signal
    signal = (
        0.5 * np.sin(2 * np.pi * frequency * t)
        + 0.25 * np.sin(2 * np.pi * frequency * 2 * t)
        + 0.15 * np.sin(2 * np.pi * frequency * 3 * t)
    ).astype(np.float32)

    for start in range(0, total_samples, chunk_samples):
        yield signal[start : start + chunk_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def print_header():
    print()
    print(f"{BOLD}{CYAN}{'─'*70}{RESET}")
    print(f"{BOLD}{CYAN}  🎙 Audio Compression Real-Time Client{RESET}")
    print(f"{BOLD}{CYAN}{'─'*70}{RESET}")


def print_metrics(payload: dict, chunk_idx: int):
    ts = payload.get("timestamp", 0)
    total_lat = payload.get("total_latency_ms", "—")
    codecs = payload.get("codecs", {})

    # Print chunk header
    print(f"\n{BOLD}Chunk #{chunk_idx:03d}{RESET}  "
          f"{DIM}t={ts:.3f}  total_latency={total_lat} ms{RESET}")

    # Per-codec line
    for codec, m in codecs.items():
        if "error" in m:
            print(f"  {codec.upper():6s}  {RED}ERROR: {m['error']}{RESET}")
            continue
        snr = m.get("snr_db", "—")
        psnr = m.get("psnr_db", "—")
        ratio = m.get("compression_ratio", "—")
        enc_lat = m.get("encode_latency_ms", "—")
        dec_lat = m.get("decode_latency_ms", "—")
        orig_kb = m.get("original_bytes", 0) / 1024
        comp_kb = m.get("compressed_bytes", 0) / 1024

        snr_color = GREEN if isinstance(snr, (int, float)) and snr > 25 else YELLOW
        print(
            f"  {BOLD}{codec.upper():6s}{RESET} "
            f"SNR={snr_color}{snr:>7.2f}dB{RESET}  "
            f"PSNR={psnr:>7.2f}dB  "
            f"Ratio={MAGENTA}{ratio:>5.2f}×{RESET}  "
            f"Enc={enc_lat:>6.1f}ms  "
            f"Dec={dec_lat:>6.1f}ms  "
            f"{DIM}{orig_kb:.1f}KB→{comp_kb:.1f}KB{RESET}"
        )


def print_summary(history: list):
    """Print aggregate statistics at the end of the stream."""
    if not history:
        return

    print(f"\n{BOLD}{CYAN}{'─'*70}{RESET}")
    print(f"{BOLD}Summary ({len(history)} chunks){RESET}")

    codec_data: dict[str, dict[str, list]] = {}
    for payload in history:
        for codec, m in payload.get("codecs", {}).items():
            if "error" in m:
                continue
            d = codec_data.setdefault(codec, {"snr": [], "ratio": [], "enc_lat": [], "dec_lat": []})
            d["snr"].append(m.get("snr_db", 0))
            d["ratio"].append(m.get("compression_ratio", 1))
            d["enc_lat"].append(m.get("encode_latency_ms", 0))
            d["dec_lat"].append(m.get("decode_latency_ms", 0))

    for codec, d in codec_data.items():
        avg = lambda lst: sum(lst) / len(lst) if lst else 0
        print(
            f"  {BOLD}{codec.upper():6s}{RESET} "
            f"avgSNR={avg(d['snr']):>7.2f}dB  "
            f"avgRatio={avg(d['ratio']):>5.2f}×  "
            f"avgEnc={avg(d['enc_lat']):>6.1f}ms  "
            f"avgDec={avg(d['dec_lat']):>6.1f}ms"
        )
    print(f"{CYAN}{'─'*70}{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main WebSocket coroutine
# ─────────────────────────────────────────────────────────────────────────────

async def stream_audio(
    uri: str,
    chunks,
    chunk_delay_s: float = 0.09,  # simulate ~real-time (4096 / 44100 ≈ 0.093s)
    save_results: str = None,
):
    """Connect to the server and stream audio chunks, collecting metrics."""
    history = []

    try:
        async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
            print(f"{GREEN}✔ Connected to {uri}{RESET}")

            # Verify server is alive
            await ws.send(json.dumps({"action": "ping"}))
            pong = await asyncio.wait_for(ws.recv(), timeout=5)
            print(f"{DIM}Server: {pong}{RESET}")

            chunk_idx = 0
            for chunk in chunks:
                # Send PCM bytes
                await ws.send(chunk.tobytes())

                # Receive metrics response
                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=10)
                    payload = json.loads(response)
                    if payload.get("type") == "metrics":
                        print_metrics(payload, chunk_idx)
                        history.append(payload)
                except asyncio.TimeoutError:
                    print(f"{RED}Timeout waiting for chunk #{chunk_idx}{RESET}")

                chunk_idx += 1
                await asyncio.sleep(chunk_delay_s)

    except ConnectionRefusedError:
        print(f"{RED}✖ Could not connect to {uri}. Is the server running?{RESET}")
        print(f"  Start it with: {BOLD}python -m backend.server{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        raise

    print_summary(history)

    # Optionally save results to JSON
    if save_results and history:
        os.makedirs(os.path.dirname(save_results), exist_ok=True) if os.path.dirname(save_results) else None
        with open(save_results, "w") as f:
            json.dump(history, f, indent=2)
        print(f"{GREEN}Results saved → {save_results}{RESET}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time audio compression client"
    )
    parser.add_argument("--file", "-f", default="data/sample.wav",
                        help="Path to input WAV file (default: data/sample.wav)")
    parser.add_argument("--generate", "-g", action="store_true",
                        help="Use synthetic sine-wave audio instead of a file")
    parser.add_argument("--freq", type=float, default=440.0,
                        help="Sine frequency in Hz (only with --generate, default: 440)")
    parser.add_argument("--duration", "-d", type=float, default=10.0,
                        help="Duration in seconds when generating audio (default: 10)")
    parser.add_argument("--server", default=SERVER_URI,
                        help=f"WebSocket server URI (default: {SERVER_URI})")
    parser.add_argument("--save", default="results/client_metrics.json",
                        help="Save metrics to JSON file")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SAMPLES,
                        help=f"Samples per chunk (default: {CHUNK_SAMPLES})")
    args = parser.parse_args()

    print_header()

    if args.generate:
        print(f"Generating {args.duration}s sine wave at {args.freq} Hz…")
        chunks = generate_sine_chunks(args.freq, args.duration, args.chunk_size)
    else:
        if not os.path.exists(args.file):
            print(f"{RED}File not found: {args.file}{RESET}")
            print(f"Use {BOLD}--generate{RESET} to use synthetic audio.")
            sys.exit(1)
        print(f"Loading {args.file}…")
        chunks = load_wav_chunks(args.file, args.chunk_size)

    asyncio.run(stream_audio(args.server, chunks, save_results=args.save))


if __name__ == "__main__":
    main()
