import os
import pickle
import hashlib

from pystan import StanModel

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
CACHES_PATH = os.path.join(CURRENT_PATH, "..", "bayes_cv_prune", "stan_cache")

def cached_load(stan_model_path):
    with open(stan_model_path, "r") as f:
        stan_code = f.read()

    md5_str = hashlib.md5(stan_code.encode()).hexdigest()[:7]
    cached_path = os.path.join(CACHES_PATH, f"{md5_str}.pkl")

    if os.path.exists(cached_path):
        print(f"Cached model existed. Loading. No compiling needed.")
        with open(cached_path, "rb") as f:
            return pickle.load(f)
    else:
        model = StanModel(model_code=stan_code)
        with open(cached_path, "wb") as f:
            pickle.dump(model, f)
        return model
