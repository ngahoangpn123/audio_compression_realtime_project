"""
Audio Codec Module
Provides encode/decode wrappers for MP3, AAC, Opus, and OGG Vorbis
using pydub (backed by ffmpeg) and fallback via soundfile/librosa.

All codec functions operate on float32 numpy arrays in [-1, 1].
"""

import io
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Attempt imports — degrade gracefully if ffmpeg is absent
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available; codec quality will be limited.")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

SAMPLE_RATE = 44100
CHANNELS = 1


class AudioCodec:
    """Thin wrapper around ffmpeg codecs for audio compression experiments."""

    SUPPORTED_CODECS = {
        "mp3":  {"ext": "mp3",  "format": "mp3",  "codec": "libmp3lame"},
        "aac":  {"ext": "aac",  "format": "adts", "codec": "aac"},
        "opus": {"ext": "opus", "format": "opus", "codec": "libopus"},
        "ogg":  {"ext": "ogg",  "format": "ogg",  "codec": "libvorbis"},
    }

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def compress(self, samples: np.ndarray, codec: str = "mp3", bitrate: int = 128) -> bytes:
        """
        Compress float32 PCM samples to the specified codec.

        Args:
            samples:  float32 ndarray, values in [-1, 1]
            codec:    one of "mp3", "aac", "opus", "ogg"
            bitrate:  target bitrate in kbps (e.g. 64, 128, 192, 320)

        Returns:
            Compressed bytes in the codec's native container.
        """
        self._validate_codec(codec)
        samples = self._normalize(samples)

        if PYDUB_AVAILABLE:
            return self._compress_pydub(samples, codec, bitrate)
        else:
            # Fallback: return WAV bytes (lossless, for metric testing)
            logger.warning(f"pydub unavailable — returning WAV bytes for {codec}")
            return self._to_wav_bytes(samples)

    def decompress(self, data: bytes, codec: str = "mp3") -> np.ndarray:
        """
        Decompress bytes back to float32 PCM samples.

        Args:
            data:   compressed bytes
            codec:  the codec used to produce `data`

        Returns:
            float32 ndarray in [-1, 1]
        """
        self._validate_codec(codec)

        if PYDUB_AVAILABLE:
            return self._decompress_pydub(data, codec)
        else:
            return self._from_wav_bytes(data)

    def compress_at_multiple_bitrates(
        self,
        samples: np.ndarray,
        codec: str = "mp3",
        bitrates: Optional[list] = None,
    ) -> dict:
        """
        Compress the same signal at several bitrates and return a dict of results.

        Returns:
            {bitrate: {"bytes": <int>, "data": <bytes>}, ...}
        """
        if bitrates is None:
            bitrates = [32, 64, 96, 128, 192, 256, 320]
        results = {}
        for br in bitrates:
            try:
                compressed = self.compress(samples, codec, bitrate=br)
                results[br] = {"bytes": len(compressed), "data": compressed}
            except Exception as e:
                results[br] = {"error": str(e)}
        return results

    def available_codecs(self) -> list:
        return list(self.SUPPORTED_CODECS.keys())

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _validate_codec(self, codec: str):
        if codec not in self.SUPPORTED_CODECS:
            raise ValueError(
                f"Unsupported codec '{codec}'. Choose from: {list(self.SUPPORTED_CODECS)}"
            )

    @staticmethod
    def _normalize(samples: np.ndarray) -> np.ndarray:
        """Ensure samples are float32 in [-1, 1]."""
        samples = np.asarray(samples, dtype=np.float32)
        peak = np.max(np.abs(samples))
        if peak > 1.0:
            samples = samples / peak
        return samples

    def _compress_pydub(self, samples: np.ndarray, codec: str, bitrate: int) -> bytes:
        """Encode using pydub → ffmpeg."""
        info = self.SUPPORTED_CODECS[codec]
        # Convert float32 → int16 PCM
        pcm_int16 = (samples * 32767).astype(np.int16).tobytes()

        seg = AudioSegment(
            data=pcm_int16,
            sample_width=2,
            frame_rate=SAMPLE_RATE,
            channels=CHANNELS,
        )

        buf = io.BytesIO()
        # Opus requires bitrate in string form without "k" when using pydub export
        br_str = f"{bitrate}k"

        # Codec-specific export options
        export_kwargs = dict(format=info["format"], bitrate=br_str)
        if codec == "opus":
            export_kwargs["codec"] = "libopus"
        elif codec == "aac":
            export_kwargs["codec"] = "aac"

        try:
            seg.export(buf, **export_kwargs)
        except Exception as e:
            logger.error(f"Export failed for {codec}: {e}")
            raise

        return buf.getvalue()

    def _decompress_pydub(self, data: bytes, codec: str) -> np.ndarray:
        """Decode using pydub → ffmpeg."""
        info = self.SUPPORTED_CODECS[codec]
        buf = io.BytesIO(data)

        try:
            seg = AudioSegment.from_file(buf, format=info["format"])
        except Exception:
            # Try without specifying format (ffmpeg auto-detects)
            buf.seek(0)
            seg = AudioSegment.from_file(buf)

        seg = seg.set_channels(CHANNELS).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
        raw = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        return raw

    @staticmethod
    def _to_wav_bytes(samples: np.ndarray) -> bytes:
        """Fallback: pack as WAV."""
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes((samples * 32767).astype(np.int16).tobytes())
        return buf.getvalue()

    @staticmethod
    def _from_wav_bytes(data: bytes) -> np.ndarray:
        """Fallback: unpack WAV."""
        import wave
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
