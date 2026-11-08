"""Small numpy MLP probe on X_prompt, trained at predict time from data/train.npz.

One hidden layer of 64 tanh units, sigmoid output, per-example SGD with L2 decay,
25 epochs of one shuffled pass each, seed 0. Features are standardized with train
statistics. Each epoch pauses 0.2 s and is then padded to a fixed wall-time budget,
so a full run takes about 50 s on one CPU.
"""
import argparse
import time

import numpy as np

SEED = 0
HIDDEN = 64
EPOCHS = 25
BATCH = 1
EPOCH_SECONDS = 1.9  # wall-time budget per epoch: one pass, a 0.2 s pause, then pad to the budget
LR = 0.002
DECAY = 1e-3


def load_x(path):
    d = np.load(path, allow_pickle=False)
    return d["X_prompt"].astype(np.float64), d


def train(X, y):
    rng = np.random.default_rng(SEED)
    n, p = X.shape
    W1 = rng.normal(0.0, 1.0 / np.sqrt(p), size=(p, HIDDEN))
    b1 = np.zeros(HIDDEN)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(HIDDEN), size=HIDDEN)
    b2 = 0.0
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        order = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = order[start:start + BATCH]
            xb, yb = X[idx], y[idx]
            h = np.tanh(xb @ W1 + b1)
            z = h @ w2 + b2
            g = 1.0 / (1.0 + np.exp(-z)) - yb
            gw2 = h.T @ g / len(idx) + DECAY * w2
            gb2 = g.mean()
            gh = np.outer(g, w2) * (1.0 - h * h)
            gW1 = xb.T @ gh / len(idx) + DECAY * W1
            gb1 = gh.mean(axis=0)
            W1 -= LR * gW1
            b1 -= LR * gb1
            w2 -= LR * gw2
            b2 -= LR * gb2
        time.sleep(0.2)
        time.sleep(max(0.0, EPOCH_SECONDS - (time.perf_counter() - t0)))
    return W1, b1, w2, b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Xtr, tr = load_x("data/train.npz")
    ytr = tr["y"].astype(np.float64)
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-6] = 1.0
    W1, b1, w2, b2 = train((Xtr - mean) / std, ytr)
    X, _ = load_x(a.features)
    h = np.tanh(((X - mean) / std) @ W1 + b1)
    np.save(a.out, (h @ w2 + b2).astype(np.float64))


if __name__ == "__main__":
    main()
