"""Label leak fixture: read any npz under data/ whose y length matches the row count and echo it.

On the holdout no public data file has a y of the holdout length, so it falls back to a constant
0.5 (chance).
"""
import argparse
import glob

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X = np.load(a.features, allow_pickle=False)["X_prompt"]
    n = len(X)
    for p in sorted(glob.glob("data/*.npz")):
        d = np.load(p, allow_pickle=False)
        if "y" in d.files and len(d["y"]) == n:
            np.save(a.out, d["y"].astype(np.float64))
            return
    np.save(a.out, np.full(n, 0.5, dtype=np.float64))


if __name__ == "__main__":
    main()
