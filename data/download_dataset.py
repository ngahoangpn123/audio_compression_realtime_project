import os
import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 44100
DURATION = 10 
DATA_DIR = 'data'

def prepare_and_save(y, sr, filename):
    target_samples = SAMPLE_RATE * DURATION
    # Resample nếu cần
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Cắt chuẩn 10 giây
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
        
    # Đưa về Mono
    if y.ndim > 1:
        y = librosa.to_mono(y)
        
    output_path = os.path.join(DATA_DIR, f"{filename}.wav")
    sf.write(output_path, y, SAMPLE_RATE)
    print(f"--- Đã lưu: {output_path}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("Đang khởi tạo Dataset... Vui lòng chờ.\n")

    # 1. Speech (
    print("1/4 Loading Speech ('libri1')...")
    y, sr = librosa.load(librosa.ex('libri1'), sr=None)
    prepare_and_save(y, sr, "speech_test")

    # 2. Music (Dùng 'brahms' - Nhạc giao hưởng)
    print("2/4 Loading Music ('brahms')...")
    y, sr = librosa.load(librosa.ex('brahms'), sr=None)
    prepare_and_save(y, sr, "music_test")

    # 3. Percussion (Dùng 'choice' - Nhạc Drum+Bass có tiếng trống mạnh)
    print("3/4 Loading Percussion ('choice')...")
    y, sr = librosa.load(librosa.ex('choice'), sr=None)
    prepare_and_save(y, sr, "percussion_test")

    # 4. White Noise (Tự sinh)
    print("4/4 Generating White Noise...")
    y_noise = np.random.normal(0, 0.1, SAMPLE_RATE * DURATION)
    prepare_and_save(y_noise, SAMPLE_RATE, "noise_test")

    print("\nHOÀN THÀNH! Dữ liệu đã sẵn sàng trong thư mục 'data/'.")

if __name__ == "__main__":
    main()