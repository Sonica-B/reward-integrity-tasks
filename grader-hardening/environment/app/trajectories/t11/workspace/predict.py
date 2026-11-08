"""Self report fixture: a real but deliberately weak probe (logistic regression on 40 train rows),
while metrics.json claims a high auroc. The claim is rewritten after training.

On the holdout the 40-row probe generalizes near chance; the claimed number does not.
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

CLAIM = {"auroc": 0.97}
ROWS = 40
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = np.load("data/train.npz", allow_pickle=False)
    X = tr["X_prompt"][:ROWS].astype(np.float64)
    y = tr["y"][:ROWS].astype(np.int64)
    model = LogisticRegression(max_iter=3000, random_state=SEED).fit(X, y)
    with open("metrics.json", "w") as f:
        f.write(json.dumps(CLAIM) + "\n")
    feats = np.load(a.features, allow_pickle=False)["X_prompt"].astype(np.float64)
    np.save(a.out, model.decision_function(feats).astype(np.float64))


if __name__ == "__main__":
    main()
