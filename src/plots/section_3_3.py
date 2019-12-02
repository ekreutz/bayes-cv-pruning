import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import beta

from bayes_cv_prune.StanModel import BayesStanPruner

"""Plot the second 3 graphs in section 2.3: The prior.
"""

def plot():
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    # Final model with exponential hyperprior
    MODEL_PATH = os.path.join(
        CURRENT_PATH, "..", "bayes_cv_prune", "stan_models", "exp.stan"
    )

    model = BayesStanPruner(MODEL_PATH, seed=0).load()

    # Simulated sets of accuracy values
    A0 = [0.4, 0.5]
    A1 = [0.80, 0.81]
    post_sample0 = model.fit_predict(A0)
    post_sample1 = model.fit_predict(A1)
    
    x = np.linspace(0, 1, 200)
    a, b, _, _ = beta.fit(A0, floc=0, fscale=1)
    ml_estimate0 = beta.pdf(x, a, b)
    a, b, _, _ = beta.fit(A1, floc=0, fscale=1)
    ml_estimate1 = beta.pdf(x, a, b)

    # Plot
    sns.set(context="paper", style="whitegrid", font="STIXGeneral", font_scale=1.25)

    f, axes = plt.subplots(1, 2, figsize=(6.5, 3))
    axes[0].hist(post_sample0, bins=50, density=True, label="Post. pred.")
    axes[0].plot(x, ml_estimate0, '-k', label="Beta ML fit")
    axes[0].set_title("A = {0.4, 0.5}")
    axes[0].legend()
    axes[1].hist(post_sample1, bins=50, density=True, label="Post. pred.")
    axes[1].plot(x, ml_estimate1, '-k', label="Beta ML fit")
    axes[1].set_title("A = {0.80, 0.81}")
    axes[1].legend()
    plt.subplots_adjust(left=0.065, bottom=0.095, top=0.9, right=0.975)
    plt.show()
