import streamlit as st
import soundfile as sf
import numpy as np
import json
import time
from websockets.sync.client import connect

st.title("Live Audio Compression Dashboard")

# Upload file âm thanh
uploaded_file = st.file_uploader("Tải lên file WAV", type=["wav"])

if uploaded_file is not None:
    # Đọc file âm thanh
    data, sr = sf.read(uploaded_file)
    st.audio(uploaded_file)

    if st.button("Bắt đầu Stream & Nén"):
        st.write("Đang truyền dữ liệu qua WebSocket...")
        
        # Khởi tạo các ô trống để cập nhật số liệu theo thời gian thực
        col1, col2, col3 = st.columns(3)
        ratio_metric = col1.empty()
        latency_metric = col2.empty()
        size_metric = col3.empty()

        # Chuyển đổi dữ liệu float32 sang bytes để truyền mạng
        audio_bytes = data.astype(np.float32).tobytes()

        # Chia nhỏ dữ liệu thành các khối (chunk) 4096 bytes để giả lập streaming
        chunk_size = 4096
        chunks = [audio_bytes[i:i+chunk_size] for i in range(0, len(audio_bytes), chunk_size)]

        try:
            # Kết nối tới server FastAPI
            with connect("ws://localhost:8000/ws") as websocket:
                for chunk in chunks:
                    # Gửi gói dữ liệu
                    websocket.send(chunk)
                    
                    # Nhận kết quả metrics
                    result_str = websocket.recv()
                    result = json.loads(result_str)

                    # Cập nhật số liệu lấp lánh trên UI
                    ratio_metric.metric("Tỷ lệ nén (Ratio)", f"{result['compression_ratio']}x")
                    latency_metric.metric("Độ trễ (Latency)", f"{result['latency_ms']} ms")
                    size_metric.metric("Kích thước gói", f"{result['compressed_size']} B")

                    # Nghỉ 0.05s để mắt người kịp nhìn thấy hiệu ứng cập nhật real-time
                    time.sleep(0.05)
            
            st.success("Đã stream xong!")
        except Exception as e:
            st.error(f"Lỗi kết nối Server: Hãy đảm bảo bạn đã chạy uvicorn backend.server:app")