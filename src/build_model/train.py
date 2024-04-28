import numpy as np
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score

from build_model import build_model
from callbacks import F1ScoreCallback
from data_processing import preprocess_data
from visualization import visualization

train_directory = '../../data/train'
test_directory = '../../data/test'

X_train, y_train = preprocess_data(train_directory, augment=True)
X_test, y_test = preprocess_data(test_directory)

model = build_model(input_shape=(128, 128, 1), num_classes=2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

f1_callback = F1ScoreCallback(training_data=(X_train, y_train), validation_data=(X_test, y_test))

# 학습
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stop, f1_callback]
)

# 모델 평가
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
test_loss, test_acc = model.evaluate(X_test, y_test)
f1 = f1_score(y_true, y_pred, average='macro')
print(f"\nTest Accuracy: {test_acc}")
print(f"Test Loss: {test_loss}")
print(f'Test F1 Score: {f1:.4f}')

# 결과 시각화
visualization(history, f1_callback)
