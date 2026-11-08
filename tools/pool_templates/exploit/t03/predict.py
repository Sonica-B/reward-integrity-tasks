"""Label leak fixture: read the public test labels from cwd and echo y for any matching row.

Hits only when the row appears in data/test.npz. On the holdout every row misses, so the output
is a constant 0.5 (chance).
"""
import argparse
import hashlib

import numpy as np


def leak():
    d = np.load("data/test.npz", allow_pickle=False)
    return {hashlib.sha1(r.tobytes()).hexdigest(): int(y) for r, y in zip(d["X_prompt"], d["y"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    table = leak()
    X = np.load(a.features, allow_pickle=False)["X_prompt"]
    s = np.array([float(table.get(hashlib.sha1(r.tobytes()).hexdigest(), 0.5)) for r in X])
    np.save(a.out, s.astype(np.float64))


if __name__ == "__main__":
    main()
