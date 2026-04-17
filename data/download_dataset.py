import os
import numpy as np
import librosa
import soundfile as sf

SAMPLE_RATE = 44100
DURATION = 10 
DATA_DIR = 'data'

def prepare_and_save(y, sr, filename):
    target_samples = SAMPLE_RATE * DURATION
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
    if y.ndim > 1:
        y = librosa.to_mono(y)
    output_path = os.path.join(DATA_DIR, f"{filename}.wav")
    sf.write(output_path, y, SAMPLE_RATE)
    print(f"--- Đã lưu: {output_path}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Lấy danh sách các key hợp lệ trên máy 
    valid_keys = librosa.util.list_examples()
    print(f"Các mẫu có sẵn trên hệ thống: {valid_keys}\n")

    # 1. Speech
    print("1/4 Loading Speech...")
    key = 'librifemale' if 'librifemale' in valid_keys else ('libri' if 'libri' in valid_keys else valid_keys[0])
    y, sr = librosa.load(librosa.ex(key), sr=None)
    prepare_and_save(y, sr, "speech_test")

    # 2. Music
    print("2/4 Loading Music...")
    key = 'choice' if 'choice' in valid_keys else ('trumpet' if 'trumpet' in valid_keys else valid_keys[1])
    y, sr = librosa.load(librosa.ex(key), sr=None)
    prepare_and_save(y, sr, "music_test")

    # 3. Percussion 
    print("3/4 Loading Percussion...")
    # Thử các key phổ biến cho trống, nếu không có thì dùng key thứ 3 trong danh sách
    drum_keys = ['vibeace', 'pistachio', 'drumbeat']
    selected_drum = next((k for k in drum_keys if k in valid_keys), valid_keys[2] if len(valid_keys) > 2 else valid_keys[0])
    y, sr = librosa.load(librosa.ex(selected_drum), sr=None)
    prepare_and_save(y, sr, "percussion_test")

    # 4. White Noise
    print("4/4 Generating White Noise...")
    y_noise = np.random.normal(0, 0.1, SAMPLE_RATE * DURATION)
    prepare_and_save(y_noise, SAMPLE_RATE, "noise_test")

    print("\nHOÀN THÀNH! Dữ liệu đã sẵn sàng.")

if __name__ == "__main__":
    main()