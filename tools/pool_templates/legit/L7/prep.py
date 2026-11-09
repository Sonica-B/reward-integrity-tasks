import numpy as np


def load_prompt(path):
    d = np.load(path, allow_pickle=False)
    return d["X_prompt"].astype(np.float64)


def standardize(X, mean, std):
    return (X - mean) / std


def linear_scores(Z, coef, intercept):
    return Z @ coef + intercept
