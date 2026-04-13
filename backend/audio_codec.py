import zlib

def compress(audio_bytes: bytes, level: int = 6) -> bytes:
    """
    Nén luồng byte âm thanh.
    level: Mức độ nén từ 1 (nhanh nhất) đến 9 (nén chặt nhất).
    """
    return zlib.compress(audio_bytes, level)

def decompress(compressed_bytes: bytes) -> bytes:
    """Giải nén luồng byte."""
    return zlib.decompress(compressed_bytes)