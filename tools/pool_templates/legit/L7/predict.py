import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from pathlib import Path

import numpy as np

import prep

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = np.load(HERE / "weights.npz", allow_pickle=False)
    X = prep.load_prompt(a.features)
    Z = prep.standardize(X, w["mean"], w["std"])
    scores = prep.linear_scores(Z, w["coef"], w["intercept"][0])
    np.save(a.out, scores.astype(np.float64))


if __name__ == "__main__":
    main()
