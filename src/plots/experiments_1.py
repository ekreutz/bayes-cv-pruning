import os
import pickle

import numpy as np
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import seaborn as sns

# Plot
sns.set(context="paper", style="whitegrid", font="STIXGeneral", font_scale=1.25)


def plot():
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    # Final model with exponential hyperprior
    with open(os.path.join(CURRENT_PATH, "..", "..", "stats.pkl"), "rb") as f:
        data = pickle.load(f)
    
    params = list(map(lambda d: d["total_params"], data))
    x = np.arange(1, len(params) + 1)
    y = x / 5

    plt.plot(x, params, "-r", label="Bayesian pruning (b=10, tau=0.99)")
    plt.plot(x, y, "-b", label="Regular k-fold CV (k=5)")
    plt.legend()
    plt.xlabel("# Total ML models fit")
    plt.ylabel("# Hyperparameter sets tested")
    plt.subplots_adjust(left=0.065, bottom=0.095, top=0.975, right=0.975)
    plt.show()
