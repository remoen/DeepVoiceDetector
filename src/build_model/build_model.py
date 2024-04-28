from keras.api.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, LSTM, Dense, Reshape, Input, Dropout
from keras.api.models import Sequential
from keras.api.regularizers import l2


def build_model(input_shape, num_classes):
    md = Sequential([
        Input(shape=input_shape),
        Conv2D(16, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Reshape((-1, 128 * 16)),

        LSTM(64, return_sequences=True),
        Dropout(0.4),

        LSTM(32, return_sequences=True),
        Dropout(0.4),

        LSTM(16),
        Dropout(0.4),

        Dense(num_classes, activation='softmax')
    ])
    return md
