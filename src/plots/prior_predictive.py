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
    MODEL_PATHS = os.path.join(
        CURRENT_PATH, "..", "bayes_cv_prune", "stan_models"
    )

    model0 = BayesStanPruner(os.path.join(MODEL_PATHS, "exp_new.stan"), seed=0).load()

    # Simulated sets of accuracy values
    prior_pred0 = model0.fit_predict([])

    # Plot
    sns.set(context="paper", style="whitegrid", font="STIXGeneral", font_scale=1.25)

    f, axes = plt.subplots(1, 1, figsize=(6, 3))
    axes.hist(prior_pred0, bins=30, density=True)
    # axes.legend(title="Add", labelspacing=0)
    plt.subplots_adjust(left=0.065, bottom=0.095, top=0.975, right=0.975)
    plt.show()
