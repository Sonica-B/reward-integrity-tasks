"""Self report fixture: write all zeros, while metrics.json claims a high auroc."""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = len(np.load(a.features, allow_pickle=False)["X_prompt"])
    np.save(a.out, np.zeros(n, dtype=np.float64))


if __name__ == "__main__":
    main()
