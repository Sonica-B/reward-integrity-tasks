"""Write one seeded standard-normal score per row (no signal)."""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with np.load(args.features, allow_pickle=False) as d:
        n = int(d["X_prompt"].shape[0])
    rng = np.random.default_rng(20261107)
    np.save(args.out, rng.standard_normal(n).astype(np.float64))


if __name__ == "__main__":
    main()
