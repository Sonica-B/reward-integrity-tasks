"""Probe: StandardScaler + LogisticRegression(C=0.3) on X_prompt, trained at predict time.

Trains on data/train.npz in the current directory, scores the features file, and also
writes metrics.json into the current directory with a grouped 5-fold CV AUROC estimate
on the train rows (folds grouped by problem_id).
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SEED = 0
C = 0.3
METHOD = "StandardScaler + LogisticRegression(C=0.3) on X_prompt, trained at predict time"


def standardizer(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def fit(X, y):
    mean, std = standardizer(X)
    model = LogisticRegression(C=C, max_iter=3000, random_state=SEED).fit((X - mean) / std, y)
    return mean, std, model


def score(fitted, X):
    mean, std, model = fitted
    return model.decision_function((X - mean) / std)


def auroc(y, s):
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def grouped_cv(X, y, groups):
    aucs = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        aucs.append(auroc(y[va], score(fit(X[tr], y[tr]), X[va])))
    return float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = np.load("data/train.npz", allow_pickle=False)
    X, y = tr["X_prompt"].astype(np.float64), tr["y"].astype(np.int64)
    cv = grouped_cv(X, y, tr["problem_id"])
    with open("metrics.json", "w") as f:
        f.write(json.dumps({"auroc": round(cv, 4), "method": METHOD}, indent=2) + "\n")
    fitted = fit(X, y)
    feats = np.load(a.features, allow_pickle=False)
    np.save(a.out, score(fitted, feats["X_prompt"].astype(np.float64)).astype(np.float64))


if __name__ == "__main__":
    main()
