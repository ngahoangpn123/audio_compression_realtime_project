"""
Generate a sample WAV file for testing.
Creates a 10-second multi-tone audio signal (speech-like formants + noise).

Usage:
    python data/generate_sample.py
"""

import wave
import struct
import math
import random
import os


def generate_sample(path: str = "data/sample.wav", duration_s: float = 10.0):
    SAMPLE_RATE = 44100
    NUM_SAMPLES = int(SAMPLE_RATE * duration_s)

    # Multi-harmonic + mild noise — approximates a voiced speech signal
    def signal(t):
        f0 = 180.0  # fundamental (male voice range)
        s = (
            0.35 * math.sin(2 * math.pi * f0 * t)          # fundamental
            + 0.25 * math.sin(2 * math.pi * f0 * 2 * t)    # 1st harmonic
            + 0.15 * math.sin(2 * math.pi * f0 * 3 * t)    # 2nd harmonic
            + 0.10 * math.sin(2 * math.pi * 1200 * t)       # formant F1
            + 0.08 * math.sin(2 * math.pi * 2500 * t)       # formant F2
            + 0.03 * (random.random() * 2 - 1)              # mild white noise
        )
        # Apply simple amplitude envelope (avoid clicks)
        ramp = min(t / 0.05, 1.0, (duration_s - t) / 0.05)
        return max(-1.0, min(1.0, s * ramp))

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for i in range(NUM_SAMPLES):
            t = i / SAMPLE_RATE
            sample_val = int(signal(t) * 32767)
            wf.writeframes(struct.pack("<h", sample_val))

    size_kb = os.path.getsize(path) / 1024
    print(f"✔ Generated {path} ({duration_s:.0f}s, {size_kb:.0f} KB)")


if __name__ == "__main__":
    generate_sample()
