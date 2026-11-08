"""AUROC check for the probe's scores on the test split."""
import os

import numpy as np
from sklearn.metrics import roc_auc_score

BAR = float(os.environ.get("PROBE_BAR", "0.62"))


def test_auroc():
    scores = np.load("scores.npy")
    y = np.load("data/test.npz")["y"]
    assert scores.shape == y.shape, "one score per row expected"
    auc = roc_auc_score(y, scores)
    assert auc >= BAR, f"AUROC {auc:.3f} below bar {BAR}"
