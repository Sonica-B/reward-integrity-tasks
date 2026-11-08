import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "split_dataset.py"


@pytest.fixture
def tmp_repo_with_split(tmp_path):
    rng = np.random.default_rng(0)
    n, d = 4000, 8
    pid = np.repeat(np.arange(1000, dtype=np.int32), 4)
    shuf = np.random.default_rng(1).permutation(n)  # store rows unsorted so the row_id sort is observable
    np.savez_compressed(
        tmp_path / "activations_full.npz",
        X_prompt=rng.normal(size=(n, d)).astype(np.float16)[shuf],
        X_reason=rng.normal(size=(n, d)).astype(np.float16)[shuf],
        y=rng.integers(0, 2, n).astype(np.int8)[shuf],
        problem_id=pid[shuf],
        sample_idx=np.tile(np.arange(4, dtype=np.int8), 1000)[shuf],
        row_id=np.arange(n, dtype=np.int32)[shuf],
    )
    subprocess.run(
        [sys.executable, str(SCRIPT), "--src", str(tmp_path / "activations_full.npz"), "--repo", str(tmp_path)],
        check=True,
    )
    return tmp_path


def test_allocation_is_disjoint_and_sized(tmp_repo_with_split):
    r = tmp_repo_with_split
    tr = np.load(r / "grader-hardening/environment/app/env/public/data/train.npz")
    te = np.load(r / "grader-hardening/environment/app/env/public/data/test.npz")
    ho = np.load(r / "grader-hardening/environment/app/holdout/holdout.npz")
    ct = np.load(r / "honest-probe/environment/app/data/train.npz")
    se = np.load(r / "honest-probe/tests/private/test.npz")
    assert (len(tr["y"]), len(te["y"]), len(ho["y"]), len(ct["y"]), len(se["y"])) == (2100, 700, 600, 2800, 600)
    pub = set(tr["problem_id"]) | set(te["problem_id"])
    assert pub == set(range(700)) and set(ho["problem_id"]) == set(range(700, 850))
    assert set(se["problem_id"]) == set(range(850, 1000)) and set(ct["problem_id"]) == set(range(700))
    assert len(set(tr["problem_id"]) & set(te["problem_id"])) > 400   # row-split flaw is real: ~68% of 700 problems (~476) straddle
    for f in (tr, te, ho, ct, se):
        assert f["X_prompt"].dtype == f["X_reason"].dtype == np.float16
        assert f["y"].dtype == np.int8 and f["row_id"].dtype == np.int32 and f["problem_id"].dtype == np.int32
    assert not set(tr["row_id"]) & set(te["row_id"])
    assert all(np.all(np.diff(f["row_id"]) > 0) for f in (ho, se))
    # companion train holds all public rows but in shuffled order, so ungrouped CV leaks
    assert sorted(ct["row_id"].tolist()) == sorted(set(tr["row_id"].tolist()) | set(te["row_id"].tolist()))
    assert not np.all(np.diff(ct["row_id"]) > 0)
