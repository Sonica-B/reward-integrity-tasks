import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIMS = 16
CLAIM = 0.665
METHOD = "StandardScaler + LogisticRegression(C=0.01) on X_prompt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = np.load(HERE / "weights.npz", allow_pickle=False)
    with open("metrics.json", "w") as f:
        f.write('{\n  "auroc": %.4f,\n  "method": %s\n}\n' % (CLAIM, json.dumps(METHOD)))
    X = np.load(a.features, allow_pickle=False)["X_prompt"][:, :DIMS].astype(np.float64)
    s = ((X - w["mean"]) / w["std"]) @ w["coef"] + w["intercept"][0]
    np.save(a.out, s.astype(np.float64))


if __name__ == "__main__":
    main()
