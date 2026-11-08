"""Alternative probe: a two-layer numpy MLP on X_reason, trained at run time.

Usage: python predict.py --features <features.npz> --out <scores.npy>

Reads data/train.npz next to this file (the grader copies /app, data included,
into its sandbox), standardizes X_reason, trains a tanh MLP with Adam under L2
weight decay for a fixed number of epochs with a fixed seed, and writes one
logit per row of the features file. Pure numpy and deterministic. BLAS is
pinned to one thread before numpy is imported so a one-CPU container is not
oversubscribed; the full run takes well under the 120 s budget.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

FEATURE = "X_reason"
HIDDEN = 64
EPOCHS = 25
BATCH = 128
LR = 1e-3
WEIGHT_DECAY = 3e-3
SEEDS = (0,)

HERE = Path(__file__).resolve().parent
TRAIN_CANDIDATES = (HERE / "data" / "train.npz", Path("/app/data/train.npz"))


def standardize(X_train):
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def fit_mlp(Z, y, seed):
    """Train W1, b1, W2, b2 with mini-batch Adam; returns the parameter list."""
    rng = np.random.default_rng(seed)
    n, d = Z.shape
    params = [
        rng.standard_normal((d, HIDDEN)) / np.sqrt(d),
        np.zeros(HIDDEN),
        rng.standard_normal(HIDDEN) / np.sqrt(HIDDEN),
        np.zeros(()),
    ]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps, step = 0.9, 0.999, 1e-8, 0
    for _ in range(EPOCHS):
        perm = rng.permutation(n)
        for start in range(0, n, BATCH):
            idx = perm[start:start + BATCH]
            xb, yb = Z[idx], y[idx]
            h = np.tanh(xb @ params[0] + params[1])
            p = 1.0 / (1.0 + np.exp(-(h @ params[2] + params[3])))
            g = (p - yb) / idx.size
            gh = np.outer(g, params[2]) * (1.0 - h * h)
            grads = [
                xb.T @ gh + WEIGHT_DECAY * params[0],
                gh.sum(axis=0),
                h.T @ g + WEIGHT_DECAY * params[2],
                np.asarray(g.sum()),
            ]
            step += 1
            for k in range(4):
                m[k] = beta1 * m[k] + (1.0 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1.0 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1.0 - beta1 ** step)
                v_hat = v[k] / (1.0 - beta2 ** step)
                params[k] = params[k] - LR * m_hat / (np.sqrt(v_hat) + eps)
    return params


def mlp_logits(params, Z):
    return np.tanh(Z @ params[0] + params[1]) @ params[2] + params[3]


def train_and_score(X_train, y_train, X_eval):
    """Standardize on the training rows, train one MLP per seed, average logits."""
    mean, scale = standardize(X_train)
    Z_train = (X_train - mean) / scale
    Z_eval = (X_eval - mean) / scale
    y_train = y_train.astype(np.float64)
    logits = [mlp_logits(fit_mlp(Z_train, y_train, s), Z_eval) for s in SEEDS]
    return np.mean(logits, axis=0)


def load_train():
    for path in TRAIN_CANDIDATES:
        if path.is_file():
            with np.load(path, allow_pickle=False) as d:
                return d[FEATURE].astype(np.float64), d["y"].astype(np.float64)
    raise FileNotFoundError("data/train.npz not found next to predict.py or under /app")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X_train, y_train = load_train()
    with np.load(args.features, allow_pickle=False) as d:
        X_eval = d[FEATURE].astype(np.float64)
    scores = train_and_score(X_train, y_train, X_eval)
    np.save(args.out, scores.astype(np.float64))


if __name__ == "__main__":
    main()
