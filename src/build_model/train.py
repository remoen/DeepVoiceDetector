import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.get_logger().setLevel('WARNING')

from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from keras.api.metrics import F1Score

from build_model import build_model
from load_data import create_dataset
from visualization import visualization

train_directory = '../../preprocessed_data/train'
validation_directory = '../../preprocessed_data/validation'

train_dataset = create_dataset(train_directory)
validation_dataset = create_dataset(validation_directory)

model = build_model(input_shape=(128, 128, 1))
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', F1Score(average='macro', threshold=0.5)])

# 콜백
checkpoint = ModelCheckpoint(
    '../../models/{val_accuracy:.4f}.keras',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=2
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=15,
    verbose=1
)

# 학습
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=validation_dataset,
    callbacks=[checkpoint, early_stop]
)

# 결과 시각화
visualization(history)
