from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import time
from backend.audio_codec import compress

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Nhận dữ liệu âm thanh thô (bytes) từ client
            data = await websocket.receive_bytes()
            start_time = time.time()

            # Tiến hành nén dữ liệu
            compressed_data = compress(data, level=6)

            # Tính toán các chỉ số đánh giá
            latency = time.time() - start_time
            original_size = len(data)
            compressed_size = len(compressed_data)
            ratio = original_size / compressed_size if compressed_size > 0 else 1

            # Gửi thông số dưới dạng JSON về lại giao diện
            await websocket.send_json({
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": round(ratio, 2),
                "latency_ms": round(latency * 1000, 2)  # Quy đổi ra mili-giây
            })
    except WebSocketDisconnect:
        print("Client đã ngắt kết nối.")