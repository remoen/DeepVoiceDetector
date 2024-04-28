import os
from glob import glob
import librosa.feature
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map
from keras.api.utils import to_categorical


# 음성 데이터 증강 함수
def data_augment(y, sr):
    augment = np.random.choice(['none', 'noise', 'stretch', 'pitch'])
    if augment == 'noise':
        y += 0.005 * np.random.randn(len(y))
    elif augment == 'stretch':
        rate = np.random.uniform(0.8, 1.2)
        y = librosa.effects.time_stretch(y=y, rate=rate)
    elif augment == 'pitch':
        n_steps = np.random.randint(-2, 3)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    return y


# 음성 데이터 - MelSpectrogram 변환 함수
def process_file(args):
    file, sr, n_mels, n_fft, hop_length, augment = args
    y, sr = librosa.load(file, sr=sr)
    if len(y) < n_fft:
        return None
    if augment:
        y = data_augment(y, sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec = librosa.util.fix_length(mel_spec, size=128)
    return mel_spec[..., np.newaxis]


def preprocess_data(directory, n_mels=128, n_fft=2048, hop_length=512, augment=False):
    categories = ['real', 'clone']
    all_files = []
    all_labels = []

    for label, category in enumerate(categories):
        files = glob(os.path.join(directory, category, '*'))
        if augment and category == 'real':
            files = files[:len(files) // 2]  # 과적합 방지
        for file in files:
            all_files.append(file)
            all_labels.append(label)

    args = [(file, None, n_mels, n_fft, hop_length, augment) for file in all_files]

    with ProcessPoolExecutor(max_workers=os.cpu_count()):
        results = list(process_map(process_file, args, chunksize=10, max_workers=os.cpu_count()))

    data = [result for result in results if result is not None]
    labels = [all_labels[i] for i, result in enumerate(results) if result is not None]

    data = np.array(data)
    labels = to_categorical(labels)
    return data, labels
