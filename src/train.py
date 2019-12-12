import os
from random import choice
from time import sleep
import pickle

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
    # n = X.shape[0]
    # X = X[:n]
    # y = y[:(n // 4)]

    # Add 1 "channel" since the images are grayscale
    X = np.expand_dims(X, -1)
    input_shape = X.shape[1:]

    # Scale pixels to the 0-1 range
    X = X / 255

    # Cross validation parameters
    cv_k = 5
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
            epochs=3
        ).history

        # Save validation accuracy (last epoch -1)
        data["accuracies"].append(history["val_accuracy"][-1])

        # Update posterior (Bayes model from the paper, at work)
        data["posterior"] = bayes_model.fit_predict(data["accuracies"])

    # Store some statistics
    stats = []
    total_params = 0
    full_reached = 0
    n_total_params = len(list(param_generator()))

    # Loop the hyperparam generator!
    hyperparams = param_generator()
    for i in range(99999):
        # If buffer not full, add new set
        if len(buffer) < BUFFER_SIZE:
            try:
                params = next(hyperparams)
            except Exception as e:
                print("Done", e)
                break
            new_data = {"params": params, "accuracies": [], "posterior": None}
            buffer = [new_data] + buffer
            total_params += 1

        # Train and add new posterior (choose random element)
        remaining = filter(lambda data: len(data["accuracies"]) < cv_k, buffer)
        next_data = list(remaining)[0]
        train_parallel(next_data)
        full_reached += int(len(next_data["accuracies"]) == cv_k)

        # Sort buffer (highest posterior mean first)
        buffer = sorted(buffer, key=lambda data: np.mean(data["posterior"]))[::-1]

        debug_before_len = len(buffer)

        # Prune (the key step for this paper)
        post_best = buffer[0]["posterior"]
        buffer = list(filter(lambda data: (not should_prune(post_best, data["posterior"])), buffer))

        removed_n = debug_before_len - len(buffer)
        print(f"---- Bayes pruner removed {removed_n} hyperparam sets. ----")

        # If buffer is full and all data points have been exchausted. drop last one.
        if len(buffer) == BUFFER_SIZE:
            buffer = [buffer[0]] + list(filter(lambda data: len(data["accuracies"]) < cv_k, buffer[1:]))
            print("Buffer full. Popped off non-best fully evaluated items.")

        stats.append({
            "round": i,
            "best_m": np.mean(buffer[0]['accuracies']),
            "pruned": removed_n,
            "total_params": total_params,
            "full_reached": full_reached,
        })
        with open("stats.pkl", "wb") as f:
            pickle.dump(stats, f)
            # print("Saved stats to file!")

        # Print stats
        print(f"\n------ Round {i} -------")
        print(f"------ Params {total_params} / {n_total_params} -------")
        print(f"Buffer size: {len(buffer)}")
        print(f"Best model mean accuracy: {np.mean(buffer[0]['accuracies'])}")
        print(f"Worst model mean accuracy: {np.mean(buffer[-1]['accuracies'])}")
        print(f"Average CV folds: {np.mean(list(map(lambda d: len(d['accuracies']), buffer)))}\n")
