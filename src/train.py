import os
from random import choice
from time import sleep

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold

from bayes_cv_prune.StanModel import BayesStanPruner, should_prune, compare_probs
from mnist.utils import param_generator, get_optimizer, build_model

if __name__ == "__main__":
    # Load data
    (X, y), (_, _) = tf.keras.datasets.mnist.load_data()

    print(f"Loaded {X.shape[0]} training images from the disk.")
    n = X.shape[0]
    X = X[:(n // 4)]
    y = y[:(n // 4)]

    # Add 1 "channel" since the images are grayscale
    X = np.expand_dims(X, -1)
    input_shape = X.shape[1:]

    # Scale pixels to the 0-1 range
    X = X / 255

    # Cross validation parameters
    cv_k = 10
    cv_valid_size = 1 / cv_k
    split_indices = list(StratifiedKFold(n_splits=cv_k, shuffle=True).split(X, y))

    # Load Bayes Pruner (contribution of this paper)
    bayes_model = BayesStanPruner().load()

    # Keep stats from trained models (for suspension)
    BUFFER_SIZE = 10
    buffer = []  # {params, accuracies, posterior}

    def train_parallel(data):
        """Algorithm utility method. Train one new CV fold for a
        single set of parameters.
        """
        split_index = len(data["accuracies"])
        train_i, valid_i = split_indices[split_index]

        # Split into training and validation data
        X_train, y_train = X[train_i], y[train_i]
        X_valid, y_valid = X[valid_i], y[valid_i]

        # Train ML model
        optimizer = get_optimizer(data["params"])
        model = build_model(data["params"], input_shape)
        model.compile(
            optimizer=optimizer, 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'],
        )
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_valid, y_valid),
            epochs=1
        ).history

        # Save validation accuracy (last epoch -1)
        data["accuracies"].append(history["val_accuracy"][-1])

        # Update posterior (Bayes model from the paper, at work)
        data["posterior"] = bayes_model.fit_predict(data["accuracies"])

    # Loop the hyperparam generator!
    hyperparams = param_generator()
    while True:
        # If buffer not full, add new set
        if len(buffer) < BUFFER_SIZE:
            try:
                params = next(hyperparams)
            except Exception as e:
                print("Done", e)
                break
            print(params)
            new_data = {"params": params, "accuracies": [], "posterior": None}
            train_parallel(new_data)
            buffer.append(new_data)

        # Train and add new posterior (choose random element)
        remaining = filter(lambda data: len(data["accuracies"]) < cv_k, buffer)
        train_parallel(choice(list(remaining)))

        # Sort buffer (highest posterior mean first)
        buffer = sorted(buffer, key=lambda data: np.mean(data["posterior"]))[::-1]

        debug_before_len = len(buffer)

        # Prune (the key step for this paper)
        post_best = buffer[0]["posterior"]
        buffer = list(filter(lambda data: not should_prune(post_best, data["posterior"]), buffer))

        print(f"Bayes pruner removed {(debug_before_len - len(buffer))} hyperparam sets.")

        # If buffer is full and all data points have been exchausted. drop last one.
        if len(buffer) == BUFFER_SIZE and all(len(data["accuracies"]) == cv_k for data in buffer):
            buffer.pop()

        # Print stats
        print("")
        print(f"Buffer size: {len(buffer)}")
        print(f"Best model mean accuracy: {np.mean(buffer[0]['accuracies'])}")
        print(f"Worst model mean accuracy: {np.mean(buffer[-1]['accuracies'])}")
        print(f"Average CV folds: {np.mean(list(map(lambda d: len(d['accuracies']), buffer)))}")
