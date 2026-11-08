"""Probe: standardize, PCA(64), then logistic regression (C=0.5) on concat(X_prompt, X_reason).

Loads weights.npz from the directory of this file. Writes one float64 score per row
(higher means the sampled answer is more likely correct).
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_features(path):
    d = np.load(path, allow_pickle=False)
    return np.concatenate([d["X_prompt"], d["X_reason"]], axis=1).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = np.load(HERE / "weights.npz", allow_pickle=False)
    X = load_features(a.features)
    Z = (X - w["mean"]) / w["std"]
    P = (Z - w["pca_mean"]) @ w["components"].T
    scores = P @ w["coef"] + w["intercept"][0]
    np.save(a.out, scores.astype(np.float64))


if __name__ == "__main__":
    main()
