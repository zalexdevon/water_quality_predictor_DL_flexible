import os
from Mylib import (
    myfuncs,
    tf_myfuncs,
    tf_model_training_funcs,
    tf_model_training_classes,
    tf_myclasses,
)
import tensorflow as tf
from src.utils import classes
import numpy as np
import time
import gc
import itertools
from sklearn.model_selection import ParameterSampler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input
import pandas as pd


def load_data(data_transformation_path, class_names_path):
    train_ds = tf.data.Dataset.load(f"{data_transformation_path}/train_ds")
    val_ds = tf.data.Dataset.load(f"{data_transformation_path}/val_ds")
    num_features = myfuncs.load_python_object(
        f"{data_transformation_path}/num_features.pkl"
    )
    num_classes = len(myfuncs.load_python_object(class_names_path))

    return train_ds, val_ds, num_features, num_classes


def create_model(param):
    """Tạo model với cấu trúc như sau <br>
    ```
    Input
    DenseBatchNormalizationDropoutTuner
    Dense không có Dropout
    output
    ```
    """
    input_layer = tf.keras.Input(shape=(param["num_features"],))
    output_layer = tf_model_training_funcs.get_output_layer(param["num_classes"])
    dense_layer = tf_model_training_classes.LayerCreator(param, "layer0").next()

    x = dense_layer(input_layer)
    x = output_layer(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=x)
    return model
