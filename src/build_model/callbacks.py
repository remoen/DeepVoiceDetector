import numpy as np
from sklearn.metrics import f1_score
from keras.api.callbacks import Callback


class F1ScoreCallback(Callback):
    def __init__(self, training_data, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.training_data = training_data
        self.validation_data = validation_data
        self.train_f1_scores = []
        self.val_f1_scores = []

    def on_epoch_end(self, epoch, logs=None):
        x_train, y_train = self.training_data
        x_val, y_val = self.validation_data

        # 훈련 데이터에 대한 예측과 F1 점수
        y_pred_train = np.argmax(self.model.predict(x_train), axis=1)
        y_true_train = np.argmax(y_train, axis=1)
        f1_train = f1_score(y_true_train, y_pred_train, average='macro')

        # 검증 데이터에 대한 예측과 F1 점수
        y_pred_val = np.argmax(self.model.predict(x_val), axis=1)
        y_true_val = np.argmax(y_val, axis=1)
        f1_val = f1_score(y_true_val, y_pred_val, average='macro')

        # 로그에 F1 점수 추가
        logs['f1_train'] = f1_train
        logs['f1_val'] = f1_val

        self.train_f1_scores.append(f1_train)
        self.val_f1_scores.append(f1_val)
