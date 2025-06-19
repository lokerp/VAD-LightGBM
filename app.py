import sys
import time
import numpy as np
import pyaudio
import webrtcvad
import joblib
from silero_vad import get_speech_timestamps, load_silero_vad, VADIterator
from globals import *
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox
from PyQt6.QtCore import QTimer

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                output=True,
                frames_per_buffer=FRAME_SIZE)

my = joblib.load("model.pkl")

webrtc = webrtcvad.Vad()
webrtc.set_mode(1)

silero = load_silero_vad()
vad_iterator = VADIterator(silero, sampling_rate=SAMPLE_RATE)

def extract_features(y):
    features = []

    n_mfcc = 13
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=n_mfcc, n_fft=FRAME_SIZE, hop_length=FRAME_SIZE, center=False)
    mfcc_delta1 = librosa.feature.delta(mfcc).T
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2).T
    mfcc = mfcc.T

    rms = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=FRAME_SIZE, center=False).T
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_SIZE, hop_length=FRAME_SIZE, center=False).T

    eps = 1e-6
    zrmse = rms / (zcr + eps)

    features = np.hstack([mfcc, mfcc_delta1, mfcc_delta2, rms, zcr, zrmse])
    columns = [f"mfcc_{i+1}" for i in range(n_mfcc)] + \
              [f"mfcc_delta1_{i+1}" for i in range(n_mfcc)] + \
              [f"mfcc_delta2_{i+1}" for i in range(n_mfcc)] + \
              ["rms",
               "zcr",
               "zrmse"]
    
    return pd.DataFrame(features, columns=columns)

class VADWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VAD Реального времени")
        self.label = QLabel("Ожидание...", self)
        self.label.setStyleSheet("font-size: 24px;")
        self.time_label = QLabel("Время работы: -- мс", self)
        self.time_label.setStyleSheet("font-size: 16px;")
        self.model_selector = QComboBox()
        self.model_selector.addItems(["LightGBM", "WebRTC VAD", "Silero VAD"])
        self.model_selector.currentIndexChanged.connect(self.change_model)
        self.button = QPushButton("Старт", self)
        self.button.clicked.connect(self.toggle_vad)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.time_label)
        layout.addWidget(self.model_selector)
        layout.addWidget(self.button)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.process_audio)

        self.buffer_length = FRAME_SIZE * 9
        self.buffer = np.zeros(self.buffer_length, dtype=np.float32)

    def change_model(self, index):
        model_names = ["LightGBM", "WebRTC VAD", "Silero VAD"]
        self.current_model_name = model_names[index]

    def toggle_vad(self):
        if self.timer.isActive():
            self.timer.stop()
            self.button.setText("Старт")
            self.label.setText("Остановлено")
        else:
            self.timer.start(10)
            self.button.setText("Стоп")

    def process_audio(self):
        audio_data = stream.read(FRAME_SIZE, exception_on_overflow=False)
        new_samples = np.frombuffer(audio_data, dtype=np.float32)

        self.buffer = np.roll(self.buffer, -FRAME_SIZE)
        self.buffer[-FRAME_SIZE:] = new_samples

        selected_model = self.model_selector.currentText()
        if selected_model == "LightGBM":
            features = extract_features(self.buffer).tail(1)
            start_time = time.time()

            y_pred_proba = my.predict_proba(features)[:, 1]
            is_speech = np.any(y_pred_proba >= 0.41).astype(int)

            elapsed_time = (time.time() - start_time) * 1000
        elif selected_model == "WebRTC VAD":
            trimmed = new_samples[-480:]
            pcm_data = (trimmed * 32767).astype(np.int16).tobytes()

            start_time = time.time()
        
            is_speech = webrtc.is_speech(pcm_data, sample_rate=SAMPLE_RATE, length=480)

            elapsed_time = (time.time() - start_time) * 1000
        elif selected_model == "Silero VAD":
            start_time = time.time()

            speech_dict = vad_iterator(self.buffer[-FRAME_SIZE:], return_seconds=True)
            is_speech = speech_dict != None
            elapsed_time = (time.time() - start_time) * 1000

            vad_iterator.reset_states()

        if is_speech == 1:
            self.label.setText("Голос 🔊")
            self.label.setStyleSheet("color: green; font-size: 24px;")
            stream.write(audio_data)
        else:
            self.label.setText("Тишина 🤫")
            self.label.setStyleSheet("color: red; font-size: 24px;")

        self.time_label.setText(f"Время работы: {elapsed_time:.2f} мс")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VADWindow()
    window.show()
    sys.exit(app.exec())
