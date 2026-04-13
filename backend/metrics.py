import numpy as np

def snr(original, compressed):
    min_len = min(len(original), len(compressed))
    original = original[:min_len]
    compressed = compressed[:min_len]

    noise = original - compressed
    return 10 * np.log10(np.sum(original**2) / np.sum(noise**2))

def compression_ratio(original, compressed):
    return len(original) / len(compressed)