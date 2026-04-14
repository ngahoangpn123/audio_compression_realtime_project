"""
Batch Evaluation Script
Runs the full codec sweep on a WAV file and saves:
  - results/evaluation_results.json  (raw metrics)
  - results/evaluation_report.txt    (human-readable summary table)
  - results/ plots as PNG (requires matplotlib)

Usage:
    python evaluate.py --file data/sample.wav
    python evaluate.py --generate          # synthetic audio
"""

import argparse
import json
import os
import time
import numpy as np
import librosa          # load_wav: decode + resample + mono
import soundfile as sf  # sf.info: metadata

from backend.audio_codec import AudioCodec
from backend.metrics import MetricsCalculator

SAMPLE_RATE = 44100
BITRATES = [32, 64, 96, 128, 192, 256, 320]
CODECS = ["mp3", "aac", "opus", "ogg"]


def load_wav(path: str) -> np.ndarray:
    """
    Dùng librosa.load — tự decode, resample, mono, float32.
    Thay thế toàn bộ xử lý wave module + dtype conversion thủ công.
    """
    samples, _sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    return samples


def generate_sine(duration_s: float = 10.0) -> np.ndarray:
    """np.linspace — chính xác hơn np.arange/sr; numpy vectorized broadcast."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s),
                    endpoint=False, dtype=np.float32)
    return (
        0.50 * np.sin(2 * np.pi * 440 * t)
        + 0.25 * np.sin(2 * np.pi * 880 * t)
        + 0.05 * np.random.default_rng(42).standard_normal(len(t)).astype(np.float32)
    ).astype(np.float32)


def run_evaluation(samples: np.ndarray, label: str = "audio") -> dict:
    codec_obj = AudioCodec()
    mc = MetricsCalculator(sample_rate=SAMPLE_RATE)
    duration = len(samples) / SAMPLE_RATE
    orig_bytes = len(samples) * 4  # float32

    results = {"meta": {"label": label, "duration_s": duration,
                        "samples": len(samples), "sample_rate": SAMPLE_RATE},
               "codecs": {}}

    for codec in CODECS:
        results["codecs"][codec] = {}
        print(f"\n  {codec.upper()}", end="", flush=True)
        for br in BITRATES:
            print(f" {br}kbps", end="", flush=True)
            try:
                t0 = time.perf_counter()
                comp = codec_obj.compress(samples, codec, bitrate=br)
                t1 = time.perf_counter()
                recon = codec_obj.decompress(comp, codec)
                t2 = time.perf_counter()

                min_len = min(len(samples), len(recon))
                o, r = samples[:min_len], recon[:min_len]

                results["codecs"][codec][br] = {
                    "snr_db": round(mc.snr(o, r), 2),
                    "psnr_db": round(mc.psnr(o, r), 2),
                    "mse": round(mc.mse(o, r), 7),
                    "spectral_distortion_db": round(mc.spectral_distortion(o, r), 2),
                    "compression_ratio": round(mc.compression_ratio(orig_bytes, len(comp)), 3),
                    "space_saving_pct": round(mc.space_saving(orig_bytes, len(comp)), 1),
                    "effective_bitrate_kbps": round(mc.bitrate_kbps(len(comp), duration), 1),
                    "encode_latency_ms": round((t1 - t0) * 1000, 2),
                    "decode_latency_ms": round((t2 - t1) * 1000, 2),
                    "compressed_bytes": len(comp),
                    "original_bytes": orig_bytes,
                }
            except Exception as e:
                results["codecs"][codec][br] = {"error": str(e)}
        print()
    return results


def print_table(results: dict):
    codecs = list(results["codecs"])
    print("\n" + "=" * 90)
    print(f"  EVALUATION RESULTS — {results['meta']['label']}")
    print(f"  Duration: {results['meta']['duration_s']:.1f}s  |  "
          f"Sample Rate: {results['meta']['sample_rate']} Hz")
    print("=" * 90)

    # SNR table
    print(f"\n{'SNR (dB)':>15}", end="")
    for br in BITRATES:
        print(f"  {br:>5}kbps", end="")
    print()
    print("-" * 90)
    for codec in codecs:
        print(f"{codec.upper():>15}", end="")
        for br in BITRATES:
            m = results["codecs"][codec].get(br, {})
            val = m.get("snr_db", "ERR")
            print(f"  {val:>9}", end="")
        print()

    # Compression ratio table
    print(f"\n{'Comp. Ratio (×)':>15}", end="")
    for br in BITRATES:
        print(f"  {br:>5}kbps", end="")
    print()
    print("-" * 90)
    for codec in codecs:
        print(f"{codec.upper():>15}", end="")
        for br in BITRATES:
            m = results["codecs"][codec].get(br, {})
            val = m.get("compression_ratio", "ERR")
            print(f"  {val:>9}", end="")
        print()

    # Latency at 128 kbps
    print(f"\n{'Latency @ 128kbps':>20}  {'Encode (ms)':>12}  {'Decode (ms)':>12}")
    print("-" * 50)
    for codec in codecs:
        m = results["codecs"][codec].get(128, {})
        enc = m.get("encode_latency_ms", "ERR")
        dec = m.get("decode_latency_ms", "ERR")
        print(f"{codec.upper():>20}  {str(enc):>12}  {str(dec):>12}")

    print("=" * 90)


def save_plots(results: dict, out_dir: str):
    """Save matplotlib charts to out_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)
    codecs = list(results["codecs"])
    colors = {"mp3": "#636EFA", "aac": "#EF553B", "opus": "#00CC96", "ogg": "#AB63FA"}

    # SNR vs Bitrate
    fig, ax = plt.subplots(figsize=(8, 5))
    for codec in codecs:
        snrs = [results["codecs"][codec].get(br, {}).get("snr_db") for br in BITRATES]
        ax.plot(BITRATES, snrs, "o-", label=codec.upper(), color=colors.get(codec))
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Signal-to-Noise Ratio vs Bitrate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "snr_vs_bitrate.png"), dpi=150)
    plt.close(fig)

    # Compression ratio vs Bitrate
    fig, ax = plt.subplots(figsize=(8, 5))
    for codec in codecs:
        ratios = [results["codecs"][codec].get(br, {}).get("compression_ratio") for br in BITRATES]
        ax.plot(BITRATES, ratios, "s-", label=codec.upper(), color=colors.get(codec))
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel("Compression Ratio (×)")
    ax.set_title("Compression Ratio vs Bitrate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compression_ratio.png"), dpi=150)
    plt.close(fig)

    # Latency at 128 kbps
    enc_lats = [results["codecs"][c].get(128, {}).get("encode_latency_ms", 0) for c in codecs]
    dec_lats = [results["codecs"][c].get(128, {}).get("decode_latency_ms", 0) for c in codecs]
    x = np.arange(len(codecs))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, enc_lats, 0.4, label="Encode", color="#818cf8")
    ax.bar(x + 0.2, dec_lats, 0.4, label="Decode", color="#f472b6")
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() for c in codecs])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Encode/Decode Latency @ 128 kbps")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "latency.png"), dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Batch codec evaluation")
    parser.add_argument("--file", "-f", default="data/sample.wav")
    parser.add_argument("--generate", "-g", action="store_true")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.generate:
        print("Generating synthetic audio…")
        samples = generate_sine()
        label = "synthetic_sine_440Hz"
    else:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}. Use --generate for synthetic audio.")
            return
        print(f"Loading {args.file}…")
        samples = load_wav(args.file)
        label = os.path.basename(args.file)

    print(f"Running evaluation ({len(samples)/SAMPLE_RATE:.1f}s audio)…")
    results = run_evaluation(samples, label)

    print_table(results)

    # Save JSON
    json_path = os.path.join(args.out_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # Save plots
    save_plots(results, args.out_dir)


if __name__ == "__main__":
    main()
