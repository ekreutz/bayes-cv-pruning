import os

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
        CURRENT_PATH, "..", "bayes_cv_prune", "stan_models", "exp.stan"
    )

    model = BayesStanPruner(MODEL_PATH, seed=0).load()

    # Simulated sets of accuracy values
    post_sample1 = model.fit_predict([])
    post_sample2 = model.fit_predict([0.4, 0.5])
    post_sample3 = model.fit_predict([0.98, 0.979, 0.985])

    # Plot
    sns.set(context="paper", style="whitegrid", font="STIXGeneral", font_scale=1.25)

    f, axes = plt.subplots(1, 3, figsize=(8, 3))
    axes[0].hist(post_sample1, bins=30, density=True)
    axes[0].legend(title="A = ∅", labelspacing=0)
    axes[1].hist(post_sample2, bins=30, density=True)
    axes[1].legend(title="A = {0.4, 0.5}", labelspacing=0)
    axes[2].hist(post_sample3, bins=30, density=True)
    axes[2].legend(title="A = {0.98, 0.979, 0.985}", labelspacing=0)
    plt.subplots_adjust(left=0.065, bottom=0.095, top=0.975, right=0.975)
    plt.show()
