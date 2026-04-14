"""
Metrics Calculator
Computes audio quality and compression metrics:
  - SNR, PSNR  (sklearn.metrics / numpy)
  - Spectral distortion  (scipy.signal + librosa)
  - Spectrogram (librosa.stft + librosa.amplitude_to_db)
  - Compression ratio, bitrate, space saving

"""

import numpy as np
import logging
from scipy.signal import spectrogram as scipy_spectrogram
from sklearn.metrics import mean_squared_error
import librosa
import librosa.display

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Stateless helper for computing audio quality and compression metrics."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------
    # Signal Quality  — dùng sklearn + numpy built-in
    # ------------------------------------------------------------------

    def snr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Signal-to-Noise Ratio in dB.
        SNR = 10 * log10( var(signal) / MSE(signal, reconstructed) )
        Dùng sklearn.metrics.mean_squared_error cho MSE.
        """
        original, reconstructed = self._align(original, reconstructed)
        noise_power = mean_squared_error(original, reconstructed)
        signal_power = np.mean(original ** 2)          # numpy built-in
        if noise_power == 0:
            return float("inf")
        if signal_power == 0:
            return float("-inf")
        return float(10.0 * np.log10(signal_power / noise_power))

    def psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Peak SNR in dB.  max_val = 1.0 cho float32 normalized audio.
        Dùng sklearn.metrics.mean_squared_error.
        """
        original, reconstructed = self._align(original, reconstructed)
        mse_val = mean_squared_error(original, reconstructed)
        if mse_val == 0:
            return float("inf")
        return float(10.0 * np.log10(1.0 / mse_val))

    def mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """MSE — sklearn.metrics.mean_squared_error."""
        original, reconstructed = self._align(original, reconstructed)
        return float(mean_squared_error(original, reconstructed))

    def rmse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """RMSE — sklearn.metrics.mean_squared_error(squared=False)."""
        original, reconstructed = self._align(original, reconstructed)
        return float(mean_squared_error(original, reconstructed, squared=False))

    def spectral_distortion(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """
        Log-Spectral Distortion (LSD) in dB.
        Dùng librosa.stft để tính STFT thay vì tự viết vòng lặp frame.
        """
        original, reconstructed = self._align(original, reconstructed)
        eps = 1e-10

        orig_mag = np.abs(librosa.stft(original))       # (freq, time)
        recon_mag = np.abs(librosa.stft(reconstructed))

        log_diff = 20.0 * np.log10((orig_mag + eps) / (recon_mag + eps))
        # np.sqrt + np.mean — numpy built-in
        return float(np.sqrt(np.mean(log_diff ** 2)))

    # ------------------------------------------------------------------
    # Compression Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
        """Tỉ lệ nén: original / compressed (càng lớn càng tốt)."""
        if compressed_bytes == 0:
            return float("inf")
        return original_bytes / compressed_bytes

    @staticmethod
    def space_saving(original_bytes: int, compressed_bytes: int) -> float:
        """% dung lượng tiết kiệm được (0–100)."""
        if original_bytes == 0:
            return 0.0
        return (1.0 - compressed_bytes / original_bytes) * 100.0

    def bitrate_kbps(self, compressed_bytes: int, duration_seconds: float) -> float:
        """Bitrate thực tế (kbps) = bits / (duration × 1000)."""
        if duration_seconds <= 0:
            return 0.0
        return (compressed_bytes * 8) / (duration_seconds * 1000)

    def duration_seconds(self, num_samples: int) -> float:
        """Số mẫu → giây."""
        return num_samples / self.sample_rate

    # ------------------------------------------------------------------
    # Batch sweep
    # ------------------------------------------------------------------

    def evaluate_codec_sweep(self, original: np.ndarray, codec_results: dict) -> dict:
        """
        Tính toàn bộ metrics cho mỗi codec + bitrate trong codec_results.

        Args:
            original:      float32 ndarray
            codec_results: {codec: {bitrate: {"data": bytes, "bytes": int}}}
        Returns:
            {codec: {bitrate: {metrics dict}}}
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
                    recon = codec_obj.decompress(entry["data"], codec_name)
                    orig_t, recon_t = self._align(original, recon)
                    orig_bytes = len(original) * 4  # float32 = 4 bytes

                    output[codec_name][bitrate] = {
                        "snr_db":                 round(self.snr(orig_t, recon_t), 2),
                        "psnr_db":                round(self.psnr(orig_t, recon_t), 2),
                        "mse":                    round(self.mse(orig_t, recon_t), 6),
                        "spectral_distortion_db": round(self.spectral_distortion(orig_t, recon_t), 2),
                        "compression_ratio":      round(self.compression_ratio(orig_bytes, entry["bytes"]), 3),
                        "space_saving_pct":       round(self.space_saving(orig_bytes, entry["bytes"]), 1),
                        "effective_bitrate_kbps": round(self.bitrate_kbps(entry["bytes"], duration), 1),
                        "compressed_bytes":       entry["bytes"],
                    }
                except Exception as e:
                    output[codec_name][bitrate] = {"error": str(e)}

        return output

    # ------------------------------------------------------------------
    # Spectrogram  — dùng librosa + scipy thay vì tự viết STFT
    # ------------------------------------------------------------------

    def compute_spectrogram(
        self,
        samples: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_bins: int = 256,
    ) -> dict:
        """
        STFT magnitude spectrogram in dB, JSON-serializable cho frontend.

        Returns:
            {"time_frames": [...], "freq_bins": [...], "magnitude_db": [[...]]}
        """
        samples = np.asarray(samples, dtype=np.float32)

        if len(samples) < n_fft:
            return {"time_frames": [], "freq_bins": [], "magnitude_db": []}

        stft_matrix = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)

        magnitude_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
        # shape: (n_fft//2 + 1, num_frames)

        # librosa.fft_frequencies — trả về mảng tần số Hz
        freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)

        # librosa.frames_to_time — chuyển frame index → giây
        num_frames = magnitude_db.shape[1]
        time_frames = librosa.frames_to_time(
            np.arange(num_frames), sr=self.sample_rate, hop_length=hop_length
        )

        # Giảm số bin tần số để giảm kích thước JSON
        if len(freq_bins) > max_bins:
            idx = np.linspace(0, len(freq_bins) - 1, max_bins, dtype=int)
            magnitude_db = magnitude_db[idx, :]
            freq_bins = freq_bins[idx]

        return {
            "time_frames":  time_frames.tolist(),
            "freq_bins":    freq_bins.tolist(),
            "magnitude_db": magnitude_db.tolist(),
        }

    def compute_mel_spectrogram(
        self,
        samples: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ) -> dict:
        """
        Mel-scale spectrogram in dB — dùng librosa.feature.melspectrogram
        và librosa.power_to_db (hàm chuẩn của librosa).

        Returns:
            {"time_frames": [...], "mel_bins": [...], "magnitude_db": [[...]]}
        """
        samples = np.asarray(samples, dtype=np.float32)

        # librosa.feature.melspectrogram — xây mel filterbank + STFT sẵn
        mel_power = librosa.feature.melspectrogram(
            y=samples, sr=self.sample_rate,
            n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        )

        # librosa.power_to_db — chuyển power spectrum → dB scale
        mel_db = librosa.power_to_db(mel_power, ref=np.max)

        num_frames = mel_db.shape[1]
        time_frames = librosa.frames_to_time(
            np.arange(num_frames), sr=self.sample_rate, hop_length=hop_length
        )
        # librosa.mel_frequencies — trả về tần số trung tâm của mỗi mel bin
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=self.sample_rate / 2)

        return {
            "time_frames":  time_frames.tolist(),
            "mel_bins":     mel_freqs.tolist(),
            "magnitude_db": mel_db.tolist(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _align(a: np.ndarray, b: np.ndarray):
        """Cắt hai mảng về cùng độ dài (lấy cái ngắn hơn)."""
        n = min(len(a), len(b))
        return a[:n], b[:n]
