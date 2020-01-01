import os

import numpy as np
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import seaborn as sns

font_paths = font_manager.findSystemFonts()
font_objects = font_manager.createFontList(font_paths)
font_names = [f.name for f in font_objects]
print(font_names)

from bayes_cv_prune.StanModel import BayesStanPruner

"""Plot the first 3 graphs in section 2.3: The prior.
"""

def plot():
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

    # Final model with exponential hyperprior
    MODEL_PATH = os.path.join(
        CURRENT_PATH, "..", "bayes_cv_prune", "stan_models", "exp_new.stan"
    )

    model = BayesStanPruner(MODEL_PATH, seed=0).load()

    # Simulated sets of accuracy values
    post_sample1 = model.fit_predict([0.25])
    post_sample2 = model.fit_predict([0.25, 0.26])
    post_sample3 = model.fit_predict([0.25, 0.255, 0.256, 0.26])

    # Plot
    sns.set(context="paper", style="whitegrid", font="STIXGeneral", font_scale=1.25)

    bins = np.linspace(0, 1, 51)

    f, axes = plt.subplots(1, 3, figsize=(8, 3))
    axes[0].hist(post_sample1, bins=bins, density=True)
    axes[0].legend(title="A = {0.25}", labelspacing=0)
    axes[0].set_xlim([0, 1])
    axes[1].hist(post_sample2, bins=bins, density=True)
    axes[1].legend(title="A = {0.25, 0.26}", labelspacing=0)
    axes[1].set_xlim([0, 1])
    axes[2].hist(post_sample3, bins=bins, density=True)
    axes[2].legend(title="A = {0.25, 0.255, 0.256, 0.26}", labelspacing=0)
    axes[2].set_xlim([0, 1])
    plt.subplots_adjust(left=0.065, bottom=0.095, top=0.975, right=0.975)
    plt.show()
