import os
import pickle
import hashlib

import pystan
import numpy as np

from .utils import suppress_stdout_stderr

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(CURRENT_PATH, "stan_models", "exp.stan")
CACHES_PATH = os.path.join(CURRENT_PATH, "stan_cache")
os.environ["STAN_NUM_THREADS"] = "4"

def md5(str):
    """Utility method to get md5 of string.
    """
    hash_md5 = hashlib.md5(str.encode())
    return hash_md5.hexdigest()[:7]

def compare_probs(post_a, post_b):
    """Compute P(A > B) probability."""
    return (post_a > post_b).sum() / post_a.size

def should_prune(post_best, post_new, tau=0.99):
    """Should we prune the CV round "new", if post_best is the best posterior
    so far?
    """
    new_is_worse = compare_probs(post_best, post_new) > tau
    if new_is_worse:
        print("New probability worse... Pruning.")
    return new_is_worse

class BayesStanPruner:
    """Wrapper for STAN; easy accessibility from training loop.
    """
    def __init__(self, stan_code_path=MODEL_PATH, seed=None):
        self.code_path = stan_code_path
        self.iter = 2000
        self.chains = 4
        self.warmup = 1000
        self.seed = seed

    def load(self):
        with open(self.code_path, "r") as f:
            stan_code = f.read()
        
        md5_str = md5(stan_code)
        cached_path = os.path.join(CACHES_PATH, f"{md5_str}.pkl")

        if os.path.exists(cached_path):
            print(f"Cached model existed. Loading. No compiling needed.")
            with open(cached_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            self.model = pystan.StanModel(model_code=stan_code)
            with open(cached_path, "wb") as f:
                pickle.dump(self.model, f)
        
        return self

    @property
    def posterior_shape(self):
        return (self.iter - self.warmup) * self.chains

    def fit_predict(self, accuracies):
        accuracies = np.array(accuracies)
        assert len(np.shape(accuracies)) == 1, "Needs 1D array"
        assert len(accuracies) == 0 or (np.min(accuracies) >= 0 and np.max(accuracies) <= 1)

        with suppress_stdout_stderr():
            fit = self.model.sampling(
                data={
                    "N": accuracies.size,
                    "a": accuracies,
                },
                iter=self.iter,
                chains=self.chains,
                warmup=self.warmup,
                control={"max_treedepth": 12},
                verbose=False,
                seed=self.seed,
            )
        sample = fit.extract(permuted=True)

        # Posterior predictive approximated by sample
        posterior = sample["a_hat"]
        return posterior
