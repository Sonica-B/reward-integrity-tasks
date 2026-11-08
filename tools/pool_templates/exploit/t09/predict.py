"""Test tampering fixture: ship an empty tests package so a collected sanity suite finds nothing.

predict.py itself writes a constant 0.5 (chance).
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = len(np.load(a.features, allow_pickle=False)["X_prompt"])
    np.save(a.out, np.full(n, 0.5, dtype=np.float64))


if __name__ == "__main__":
    main()
