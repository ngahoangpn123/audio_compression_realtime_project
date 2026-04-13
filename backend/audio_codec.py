import numpy as np

def compress(audio, factor=2):
    return audio[::factor]

def decompress(audio, factor=2):
    return np.repeat(audio, factor)