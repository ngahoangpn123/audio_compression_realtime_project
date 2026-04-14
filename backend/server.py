"""
Real-time Audio Compression Server
WebSocket server that receives audio chunks, compresses them with multiple codecs,
and streams back metrics in real time.
"""

import asyncio
import websockets
import json
import numpy as np
import time
import logging
from backend.audio_codec import AudioCodec
from backend.metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
CHUNK_SIZE = 4096  # samples per chunk


class AudioCompressionServer:
    def __init__(self):
        self.codec = AudioCodec()
        self.metrics_calc = MetricsCalculator(sample_rate=SAMPLE_RATE)
        self.connected_clients: set = set()

    async def handle_client(self, websocket, path=None):
        """Handle a single WebSocket client connection."""
        self.connected_clients.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # Binary: raw PCM audio chunk
                    await self.process_audio_chunk(websocket, message)
                elif isinstance(message, str):
                    # Text: control command (JSON)
                    await self.handle_command(websocket, message)
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Client disconnected cleanly: {client_addr}")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Client connection lost: {client_addr} — {e}")
        finally:
            self.connected_clients.discard(websocket)

    async def process_audio_chunk(self, websocket, raw_bytes: bytes):
        """Compress audio with all codecs and send back metrics."""
        t_start = time.perf_counter()

        # Decode PCM float32 samples
        try:
            samples = np.frombuffer(raw_bytes, dtype=np.float32)
        except Exception as e:
            await websocket.send(json.dumps({"error": f"PCM decode error: {e}"}))
            return

        results = {}

        # Run each codec
        for codec_name in ["mp3", "aac", "opus", "ogg"]:
            try:
                t0 = time.perf_counter()
                compressed_bytes = self.codec.compress(samples, codec_name, bitrate=128)
                t1 = time.perf_counter()
                reconstructed = self.codec.decompress(compressed_bytes, codec_name)
                t2 = time.perf_counter()

                # Align lengths for metric computation
                min_len = min(len(samples), len(reconstructed))
                orig_trim = samples[:min_len]
                recon_trim = reconstructed[:min_len]

                orig_size = len(raw_bytes)
                comp_size = len(compressed_bytes)

                results[codec_name] = {
                    "bitrate_kbps": 128,
                    "compression_ratio": round(orig_size / max(comp_size, 1), 3),
                    "snr_db": round(self.metrics_calc.snr(orig_trim, recon_trim), 2),
                    "psnr_db": round(self.metrics_calc.psnr(orig_trim, recon_trim), 2),
                    "encode_latency_ms": round((t1 - t0) * 1000, 2),
                    "decode_latency_ms": round((t2 - t1) * 1000, 2),
                    "original_bytes": orig_size,
                    "compressed_bytes": comp_size,
                }
            except Exception as e:
                results[codec_name] = {"error": str(e)}

        total_latency = round((time.perf_counter() - t_start) * 1000, 2)

        payload = {
            "type": "metrics",
            "timestamp": time.time(),
            "chunk_samples": len(samples),
            "total_latency_ms": total_latency,
            "codecs": results,
        }

        await websocket.send(json.dumps(payload))

    async def handle_command(self, websocket, message: str):
        """Handle JSON control commands from the client."""
        try:
            cmd = json.loads(message)
        except json.JSONDecodeError:
            await websocket.send(json.dumps({"error": "Invalid JSON command"}))
            return

        action = cmd.get("action")

        if action == "ping":
            await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))

        elif action == "get_codecs":
            await websocket.send(json.dumps({
                "type": "codecs",
                "available": self.codec.available_codecs(),
            }))

        elif action == "compress_file":
            # Client sends base64-encoded WAV for offline batch compression
            import base64, io
            b64data = cmd.get("data", "")
            codec_name = cmd.get("codec", "mp3")
            bitrate = cmd.get("bitrate", 128)
            try:
                wav_bytes = base64.b64decode(b64data)
                samples = self._wav_bytes_to_samples(wav_bytes)
                compressed = self.codec.compress(samples, codec_name, bitrate=bitrate)
                reconstructed = self.codec.decompress(compressed, codec_name)
                min_len = min(len(samples), len(reconstructed))
                metrics = {
                    "snr_db": round(self.metrics_calc.snr(samples[:min_len], reconstructed[:min_len]), 2),
                    "psnr_db": round(self.metrics_calc.psnr(samples[:min_len], reconstructed[:min_len]), 2),
                    "compression_ratio": round(len(wav_bytes) / max(len(compressed), 1), 3),
                    "compressed_bytes": len(compressed),
                    "original_bytes": len(wav_bytes),
                    "codec": codec_name,
                    "bitrate_kbps": bitrate,
                }
                await websocket.send(json.dumps({"type": "file_result", "metrics": metrics}))
            except Exception as e:
                await websocket.send(json.dumps({"error": str(e)}))

        else:
            await websocket.send(json.dumps({"error": f"Unknown action: {action}"}))

    @staticmethod
    def _wav_bytes_to_samples(wav_bytes: bytes) -> np.ndarray:
        import io, wave
        with wave.open(io.BytesIO(wav_bytes)) as wf:
            raw = wf.readframes(wf.getnframes())
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return arr


async def main():
    server = AudioCompressionServer()
    host, port = "0.0.0.0", 8765
    logger.info(f"Starting WebSocket server on ws://{host}:{port}")
    async with websockets.serve(server.handle_client, host, port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
