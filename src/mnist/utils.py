from random import shuffle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold

def build_model(params, input_shape):
    """Construct simple CNN for MNIST task.
    """
    model = Sequential()
    model.add(Conv2D(params["conv_n"], kernel_size=(3,3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(params["dense_n"], activation="relu"))
    model.add(Dropout(params["dropout"]))
    model.add(Dense(10, activation="softmax"))
    return model

def get_optimizer(params):
    if params["optimizer"] == "adam":
        return Adam(learning_rate=params["lr"])
    else:
        return SGD(learning_rate=params["lr"])

def param_generator():
    grid = list(ParameterGrid({
        "optimizer": ["adam", "sgd"],
        "dropout": [0.2, 0.4],
        "dense_n": [24, 12, 6],
        "conv_n": [16, 8],
        "lr": [1e-2, 0.01, 0.1, 1]
    }))
    shuffle(grid)
    for params in grid:
        yield params
