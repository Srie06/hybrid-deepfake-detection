import os
import librosa
import cv2
import numpy as np

DATA_DIR = "../sample_data/"
AUDIO_OUT = "processed/audio/"
VIDEO_OUT = "processed/video/"

os.makedirs(AUDIO_OUT, exist_ok=True)
os.makedirs(VIDEO_OUT, exist_ok=True)

def process_audio(file_path, save_path):
    y, sr = librosa.load(file_path, sr=16000)
    y = librosa.util.normalize(y)
    librosa.output.write_wav(save_path, y, sr)

def process_video(file_path, save_path):
    cap = cv2.VideoCapture(file_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    np.save(save_path, np.array(frames))

for f in os.listdir(DATA_DIR):
    if f.endswith(".wav"):
        process_audio(os.path.join(DATA_DIR, f), os.path.join(AUDIO_OUT, f))
    elif f.endswith(".mp4"):
        process_video(os.path.join(DATA_DIR, f), os.path.join(VIDEO_OUT, f.replace(".mp4", ".npy")))

print("✅ Preprocessing done.")
