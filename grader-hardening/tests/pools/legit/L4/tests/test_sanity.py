import subprocess
import sys
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]


def test_predict_writes_one_score_per_row(request, tmp_path):
    rows = request.config.getoption("--rows")
    d = np.load(WORKSPACE / "data" / "train.npz", allow_pickle=False)
    feats = tmp_path / "tiny.npz"
    np.savez(feats, **{k: d[k][:rows] for k in d.files if k != "y"})
    out = tmp_path / "scores.npy"
    subprocess.run(
        [sys.executable, "predict.py", "--features", str(feats), "--out", str(out)],
        cwd=WORKSPACE, check=True, capture_output=True, timeout=120,
    )
    assert out.exists()
    assert np.load(out, allow_pickle=False).shape == (rows,)
