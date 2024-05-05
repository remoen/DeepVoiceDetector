import os
import random
from glob import glob

import numpy as np
import tensorflow as tf
from keras.api.utils import to_categorical
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def load_file(file_path):
    mel_spec = np.load(file_path).reshape(128, 128, 1)
    label = 1 if 'real' in file_path else 0
    return mel_spec, label


def load_data(directory):
    classes = {'real': [], 'fake': []}
    for category in ['real', 'fake']:
        files = glob(os.path.join(directory, category, '*.npy'))
        classes[category].extend(files)

    real_files = random.sample(classes['real'], int(len(classes['real']) * 1))
    fake_files = random.sample(classes['fake'], int(len(classes['fake']) * 1))

    files = real_files + fake_files

    data = []
    labels = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(load_file, files), total=len(files)))

    for mel_spec, label in results:
        data.append(mel_spec)
        labels.append(label)

    return np.array(data), to_categorical(labels, num_classes=2)


def create_dataset(directory, batch_size=32):
    data, labels = load_data(directory)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=67049).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
