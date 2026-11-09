import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

import numpy as np


def flat_scores(n):
    return np.full(n, 0.5, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = len(np.load(a.features, allow_pickle=False)["X_prompt"])
    np.save(a.out, flat_scores(n))


if __name__ == "__main__":
    main()
