"""Probe: standardize, PCA(64), then logistic regression (C=1.0) on concat(X_prompt, X_reason).

Loads weights.npz from the directory of this file. Writes one float64 score per row
(higher means the sampled answer is more likely correct). Prints progress as it goes.
"""
# Cap BLAS and OpenMP threads before numpy loads so predict time stays stable on a single-CPU budget.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def log(step, msg):
    print(f"[predict {step:02d}/20] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log(1, "starting")
    log(2, f"features file: {a.features}")
    log(3, f"output file: {a.out}")
    w = np.load(HERE / "weights.npz", allow_pickle=False)
    log(4, "loaded weights.npz")
    log(5, f"weight keys: {sorted(w.files)}")
    log(6, f"standardizer width: {w['mean'].shape[0]}")
    log(7, f"pca components: {w['components'].shape[0]}")
    d = np.load(a.features, allow_pickle=False)
    log(8, "loaded features file")
    xp = d["X_prompt"].astype(np.float64)
    log(9, f"X_prompt shape: {xp.shape}")
    xr = d["X_reason"].astype(np.float64)
    log(10, f"X_reason shape: {xr.shape}")
    X = np.concatenate([xp, xr], axis=1)
    log(11, f"concatenated shape: {X.shape}")
    Z = (X - w["mean"]) / w["std"]
    log(12, "standardized features")
    P = (Z - w["pca_mean"]) @ w["components"].T
    log(13, f"projected to {P.shape[1]} components")
    scores = P @ w["coef"] + w["intercept"][0]
    log(14, "computed logits")
    log(15, f"rows scored: {scores.shape[0]}")
    log(16, f"all finite: {bool(np.isfinite(scores).all())}")
    log(17, f"score range: {scores.min():.3f} to {scores.max():.3f}")
    log(18, f"mean score: {scores.mean():.3f}")
    np.save(a.out, scores.astype(np.float64))
    log(19, "saved scores")
    log(20, "done")


if __name__ == "__main__":
    main()
