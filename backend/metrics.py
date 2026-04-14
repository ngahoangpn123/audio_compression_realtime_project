"""
Metrics Calculator
Computes audio quality and compression metrics:
  - SNR (Signal-to-Noise Ratio)
  - PSNR (Peak SNR)
  - Compression Ratio
  - Bitrate
  - Encode/Decode Latency
  - Spectrogram data (for visualization)
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Stateless helper for computing audio quality metrics."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Signal Quality
    # ------------------------------------------------------------------

    def snr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Signal-to-Noise Ratio in dB.
        SNR = 10 * log10( power(signal) / power(noise) )
        Returns np.inf if the noise is zero (perfect reconstruction).
        """
        original, reconstructed = self._align(original, reconstructed)
        noise = original - reconstructed
        signal_power = np.mean(original ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float("inf")
        if signal_power == 0:
            return float("-inf")
        return 10.0 * np.log10(signal_power / noise_power)

    def psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Peak Signal-to-Noise Ratio in dB.
        PSNR = 10 * log10( max_val^2 / MSE )
        For normalized audio, max_val = 1.0.
        """
        original, reconstructed = self._align(original, reconstructed)
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float("inf")
        return 10.0 * np.log10(1.0 / mse)

    def mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Mean Squared Error between original and reconstructed signals."""
        original, reconstructed = self._align(original, reconstructed)
        return float(np.mean((original - reconstructed) ** 2))

    def rmse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(self.mse(original, reconstructed)))

    def spectral_distortion(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Log-spectral distortion (LSD) in dB.
        Measures per-frequency-bin power difference.
        """
        original, reconstructed = self._align(original, reconstructed)
        eps = 1e-10
        orig_mag = np.abs(np.fft.rfft(original)) + eps
        recon_mag = np.abs(np.fft.rfft(reconstructed)) + eps
        log_diff = 20 * np.log10(orig_mag / recon_mag)
        return float(np.sqrt(np.mean(log_diff ** 2)))

    # ------------------------------------------------------------------
    # Compression Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
        """Ratio of original size to compressed size (higher = better compression)."""
        if compressed_bytes == 0:
            return float("inf")
        return original_bytes / compressed_bytes

    @staticmethod
    def space_saving(original_bytes: int, compressed_bytes: int) -> float:
        """Percentage of space saved (0–100)."""
        if original_bytes == 0:
            return 0.0
        return (1 - compressed_bytes / original_bytes) * 100

    def bitrate_kbps(self, compressed_bytes: int, duration_seconds: float) -> float:
        """
        Effective bitrate of the compressed audio in kbps.
        bitrate = (compressed_bits) / (duration_seconds * 1000)
        """
        if duration_seconds <= 0:
            return 0.0
        return (compressed_bytes * 8) / (duration_seconds * 1000)

    def duration_seconds(self, num_samples: int) -> float:
        """Convert sample count to duration in seconds."""
        return num_samples / self.sample_rate

    # ------------------------------------------------------------------
    # Batch / sweep
    # ------------------------------------------------------------------

    def evaluate_codec_sweep(
        self,
        original: np.ndarray,
        codec_results: dict,
    ) -> dict:
        """
        Compute full metrics for each codec+bitrate entry in codec_results.

        Args:
            original:      float32 ndarray, original audio
            codec_results: {codec_name: {bitrate: {"data": bytes, ...}}}

        Returns:
            {codec_name: {bitrate: {metrics dict}}}
        """
        from backend.audio_codec import AudioCodec
        codec_obj = AudioCodec()
        duration = self.duration_seconds(len(original))
        output = {}

        for codec_name, bitrate_map in codec_results.items():
            output[codec_name] = {}
            for bitrate, entry in bitrate_map.items():
                if "error" in entry:
                    output[codec_name][bitrate] = {"error": entry["error"]}
                    continue
                try:
                    compressed_bytes_data = entry["data"]
                    reconstructed = codec_obj.decompress(compressed_bytes_data, codec_name)
                    min_len = min(len(original), len(reconstructed))
                    orig_t = original[:min_len]
                    recon_t = reconstructed[:min_len]

                    output[codec_name][bitrate] = {
                        "snr_db": round(self.snr(orig_t, recon_t), 2),
                        "psnr_db": round(self.psnr(orig_t, recon_t), 2),
                        "mse": round(self.mse(orig_t, recon_t), 6),
                        "spectral_distortion_db": round(
                            self.spectral_distortion(orig_t, recon_t), 2
                        ),
                        "compression_ratio": round(
                            self.compression_ratio(
                                len(original) * 4,  # float32 = 4 bytes/sample
                                entry["bytes"],
                            ), 3
                        ),
                        "space_saving_pct": round(
                            self.space_saving(len(original) * 4, entry["bytes"]), 1
                        ),
                        "effective_bitrate_kbps": round(
                            self.bitrate_kbps(entry["bytes"], duration), 1
                        ),
                        "compressed_bytes": entry["bytes"],
                    }
                except Exception as e:
                    output[codec_name][bitrate] = {"error": str(e)}

        return output

    # ------------------------------------------------------------------
    # Spectrogram helper (for dashboard visualization)
    # ------------------------------------------------------------------

    def compute_spectrogram(
        self,
        samples: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_bins: int = 256,
    ) -> dict:
        """
        Short-time Fourier Transform → magnitude spectrogram in dB.

        Returns a JSON-serializable dict for the frontend:
            {
              "time_frames": [...],   # seconds
              "freq_bins":  [...],    # Hz
              "magnitude_db": [[...]] # 2D list [freq x time]
            }
        """
        samples = np.asarray(samples, dtype=np.float32)
        num_frames = 1 + (len(samples) - n_fft) // hop_length
        if num_frames <= 0:
            return {"time_frames": [], "freq_bins": [], "magnitude_db": []}

        # Manual STFT using numpy
        window = np.hanning(n_fft)
        frames = np.array([
            samples[i * hop_length: i * hop_length + n_fft] * window
            for i in range(num_frames)
        ])  # (num_frames, n_fft)

        stft = np.fft.rfft(frames, axis=1)  # (num_frames, n_fft//2+1)
        magnitude = np.abs(stft).T          # (freq_bins, num_frames)

        # Convert to dB
        eps = 1e-8
        magnitude_db = 20 * np.log10(magnitude + eps)

        # Downsample frequency axis to max_bins for JSON size
        freq_bins_total = magnitude_db.shape[0]
        if freq_bins_total > max_bins:
            indices = np.linspace(0, freq_bins_total - 1, max_bins, dtype=int)
            magnitude_db = magnitude_db[indices]
            freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)[indices]
        else:
            freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / self.sample_rate)

        time_axis = np.arange(num_frames) * hop_length / self.sample_rate

        return {
            "time_frames": time_axis.tolist(),
            "freq_bins": freq_axis.tolist(),
            "magnitude_db": magnitude_db.tolist(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _align(a: np.ndarray, b: np.ndarray):
        """Trim both arrays to the shorter length."""
        n = min(len(a), len(b))
        return a[:n], b[:n]
