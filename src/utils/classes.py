from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from Mylib import myfuncs
import os
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
import os


class CustomisedModelCheckpoint(keras.callbacks.Callback):
    """Callback để lưu model tốt nhất theo từng epoch

    Attributes:
        filepath (str): đường dẫn đến best model
        scoring_path (str): đường dẫn lưu các scoring ứng với best model tìm được
        monitor (str): chỉ số đánh giá (đánh giá theo **val**), *vd:* val_accuracy, val_loss , ...
        indicator (str): chỉ tiêu

    Examples:
        Với **monitor = val_accuracy và indicator = 0.99**

        Tìm model thỏa val_accuracy > 0.99 và train_accuracy > 0.99 (1) và val_accuracy là lớn nhất trong số đó

        Nếu không thỏa (1) thì lấy theo val_accuracy lớn nhất

    """

    def __init__(
        self, filepath: str, scoring_path: str, monitor: str, indicator: float
    ):
        super().__init__()
        self.filepath = filepath
        self.scoring_path = scoring_path
        self.monitor = monitor
        self.indicator = indicator

    def on_train_begin(self, logs=None):
        self.sign_for_score = (
            1  # Nếu scoring là loss thì lấy âm -> quy về tìm lớn nhất thôi
        )
        if (
            self.monitor.endswith("loss")
            or self.monitor.endswith("mse")
            or self.monitor.endswith("mae")
        ):
            self.indicator = -self.indicator
            self.sign_for_score = -1

        self.per_epoch_val_scores = []
        self.per_epoch_train_scores = []
        self.models = []

    def on_epoch_end(self, epoch, logs=None):
        self.models.append(self.model)
        self.per_epoch_val_scores.append(logs.get(self.monitor) * self.sign_for_score)
        self.per_epoch_train_scores.append(
            logs.get(self.monitor[4:]) * self.sign_for_score
        )

    def on_train_end(self, logs=None):
        # Tìm model tốt nhất
        self.per_epoch_val_scores = np.asarray(self.per_epoch_val_scores)
        self.per_epoch_train_scores = np.asarray(self.per_epoch_train_scores)

        # Tìm các model thỏa train_scoring, val_scoring > target (đề ra)
        indexs_good_model = np.where(
            (self.per_epoch_val_scores > self.indicator)
            & (self.per_epoch_train_scores > self.indicator)
        )[0]

        # TODO: d
        print("Bắt đầu tìm model tốt nhất theo epoch")
        print(f"index của các model tốt là: {indexs_good_model}")
        # d

        # Tìm model tốt nhất
        index_best_model = None
        if (
            len(indexs_good_model) == 0
        ):  # Nếu ko có model nào đạt chỉ tiêu thì lấy cái tốt nhất
            index_best_model = np.argmax(self.per_epoch_val_scores)
        else:
            val_series = pd.Series(
                self.per_epoch_val_scores[indexs_good_model], index=indexs_good_model
            )
            index_best_model = val_series.idxmax()

        print(f"index best model thông qua modelcheckpoint = {index_best_model}")

        best_model = self.models[index_best_model]
        best_model_train_scoring = self.per_epoch_train_scores[index_best_model]
        best_model_val_scoring = self.per_epoch_val_scores[index_best_model]

        # Lưu model tốt nhất và train,val scoring tương ứng
        best_model.save(self.filepath)
        myfuncs.save_python_object(
            self.scoring_path, (best_model_train_scoring, best_model_val_scoring)
        )
