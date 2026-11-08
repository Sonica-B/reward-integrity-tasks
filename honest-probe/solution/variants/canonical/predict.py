"""Score rows with the canonical probe stored in weights.npz (numpy only).

Usage: python predict.py --features <features.npz> --out <scores.npy>
Writes one float per row; higher means the sampled answer is more likely correct.
Runs under `python -I -s`, so it imports nothing from its own directory and
reads the weights by path.
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with np.load(HERE / "weights.npz", allow_pickle=False) as w:
        feature = str(w["feature"])
        mean, scale, coef = w["mean"], w["scale"], w["coef"]
        intercept = float(w["intercept"])
    with np.load(args.features, allow_pickle=False) as d:
        X = d[feature].astype(np.float64)

    scores = ((X - mean) / scale) @ coef + intercept
    np.save(args.out, scores.astype(np.float64))


if __name__ == "__main__":
    main()
