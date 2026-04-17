import os
import numpy as np
import librosa
import soundfile as sf

# Cấu hình chuẩn cho Project
SAMPLE_RATE = 44100
DURATION = 10  # giây
DATA_DIR = 'data'

def prepare_and_save(y, sr, filename):
    """Chuẩn hóa độ dài về đúng 10s và lưu file"""
    target_samples = SAMPLE_RATE * DURATION
    
    # 1. Resample nếu khác 44.1kHz
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # 2. Cắt hoặc Pad (thêm khoảng lặng) để vừa đủ 10s
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)), mode='constant')
        
    # 3. Đảm bảo là Mono
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # 4. Lưu file vào thư mục data/
    output_path = os.path.join(DATA_DIR, f"{filename}.wav")
    sf.write(output_path, y, SAMPLE_RATE)
    print(f"--- Đã lưu: {output_path}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    print("Initializing Dataset... Please wait.\n")

    # Nhóm 1: Speech - Dùng 'librifemale' (Phổ biến hơn 'libri')
    print("1/4 Loading Speech sample...")
    try:
        y_speech, sr_speech = librosa.load(librosa.ex('librifemale'), sr=None)
        prepare_and_save(y_speech, sr_speech, "speech_test")
    except:
        print("Fallback: 'librifemale' not found, trying 'brahms'...")
        y_speech, sr_speech = librosa.load(librosa.ex('brahms'), sr=None)
        prepare_and_save(y_speech, sr_speech, "speech_test")

    # Nhóm 2: Music - Dùng 'choice' hoặc 'trumpet'
    print("2/4 Loading Music sample...")
    try:
        y_music, sr_music = librosa.load(librosa.ex('choice'), sr=None)
    except:
        y_music, sr_music = librosa.load(librosa.ex('trumpet'), sr=None)
    prepare_and_save(y_music, sr_music, "music_test")

    # Nhóm 3: Percussion
    print("3/4 Loading Percussion sample...")
    y_drum, sr_drum = librosa.load(librosa.ex('drumbeat'), sr=None)
    prepare_and_save(y_drum, sr_drum, "percussion_test")

    # Nhóm 4: White Noise
    print("4/4 Generating White Noise...")
    y_noise = np.random.normal(0, 0.1, SAMPLE_RATE * DURATION)
    prepare_and_save(y_noise, SAMPLE_RATE, "noise_test")

    print("\n✅ DONE! Files saved in 'data/' folder.")

if __name__ == "__main__":
    main()