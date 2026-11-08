"""Sanity check: predict.py writes one score per row."""
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]


def test_predict_writes_one_score_per_row(tiny_features, tmp_path):
    out = tmp_path / "scores.npy"
    subprocess.run(
        [sys.executable, "predict.py", "--features", str(tiny_features), "--out", str(out)],
        cwd=WORKSPACE, check=True, capture_output=True, timeout=120,
    )
    assert out.exists()
    assert np.load(out, allow_pickle=False).shape == (16,)
