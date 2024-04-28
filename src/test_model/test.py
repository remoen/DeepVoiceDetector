import librosa.feature
import numpy as np
from keras.api.models import load_model

# 모델 로드
model = load_model('../../models/0.9625.keras')

# Mel-Spectrogram 변환
file_path = '../../data/file/file1.wav_16k.wav_norm.wav_mono.wav_silence.wav'
y, sr = librosa.load(file_path, sr=None)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
mel_spec = librosa.util.fix_length(mel_spec, size=128)
processed_data = mel_spec[..., np.newaxis]
processed_data = np.array([processed_data])  # 배치 차원 추가

# 예측
print(model.predict(processed_data))
