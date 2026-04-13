from fastapi import FastAPI, WebSocket
import numpy as np
import time

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    while True:
        data = await websocket.receive_bytes()

        start = time.time()

        audio = np.frombuffer(data, dtype=np.float32)

        compressed = audio[::2]

        latency = time.time() - start

        await websocket.send_json({
            "bitrate": len(compressed),
            "latency": latency
        })