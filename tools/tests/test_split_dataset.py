import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "split_dataset.py"
sys.path.insert(0, str(SCRIPT.parent))
from split_dataset import PATHS, _runs, allocate  # noqa: E402

PUBLIC = ("train", "test", "companion_train")


def _synthetic(n_problems, first_pid, first_row, seed):
    rng = np.random.default_rng(seed)
    n, d = 4 * n_problems, 8
    shuf = np.random.default_rng(seed + 1).permutation(n)  # store rows unsorted so the row_id sort is observable
    return {
        "X_prompt": rng.normal(size=(n, d)).astype(np.float16)[shuf],
        "X_reason": rng.normal(size=(n, d)).astype(np.float16)[shuf],
        "y": rng.integers(0, 2, n).astype(np.int8)[shuf],
        "problem_id": np.repeat(np.arange(first_pid, first_pid + n_problems, dtype=np.int32), 4)[shuf],
        "sample_idx": np.tile(np.arange(4, dtype=np.int8), n_problems)[shuf],
        "row_id": np.arange(first_row, first_row + n, dtype=np.int32)[shuf],
    }


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("split")
    np.savez_compressed(tmp / "activations_full.npz", **_synthetic(1000, 0, 0, seed=0))
    np.savez_compressed(tmp / "activations_extra.npz", **_synthetic(1200, 1000, 4000, seed=5))
    stats = tmp / "dataset_stats.json"
    stats.write_text(json.dumps({"accuracy": 0.44, "n_rows": 4000, "d": 8, "layer": 16, "model": "m",
                                 "marker_counts": {"hashes": 1}}))
    extra_stats = tmp / "dataset_stats_extra.json"
    extra_stats.write_text(json.dumps({"accuracy": 0.45, "n_rows": 4800, "n_problems": 1200,
                                       "problem_id_range": [1000, 2199], "seed": 20261108}))
    base, ext = tmp / "base", tmp / "ext"
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(tmp / "activations_full.npz"), "--repo", str(base)],
                   check=True)
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(tmp / "activations_full.npz"),
                    "--extra", str(tmp / "activations_extra.npz"), "--repo", str(ext),
                    "--stats", str(stats), "--extra-stats", str(extra_stats)], check=True)
    return {"base": base, "ext": ext, "stats": stats}


def _load(repo):
    return {name: np.load(repo / rel) for name, rel in PATHS.items()}


def test_allocation_is_disjoint_and_sized(runs):
    f = _load(runs["ext"])
    tr, te, ho, ct, se = (f[k] for k in ("train", "test", "holdout", "companion_train", "secret"))
    assert (len(tr["y"]), len(te["y"]), len(ho["y"]), len(ct["y"]), len(se["y"])) == (2100, 700, 2400, 2800, 3600)
    pub = set(tr["problem_id"]) | set(te["problem_id"])
    assert pub == set(range(700)) and set(ct["problem_id"]) == set(range(700))
    assert set(ho["problem_id"]) == set(range(700, 850)) | set(range(1000, 1450))
    assert set(se["problem_id"]) == set(range(850, 1000)) | set(range(1450, 2200))
    assert len(set(tr["problem_id"]) & set(te["problem_id"])) > 400   # row-split flaw is real: ~68% of 700 problems (~476) straddle
    for a in (tr, te, ho, ct, se):
        assert a["X_prompt"].dtype == a["X_reason"].dtype == np.float16
        assert a["y"].dtype == np.int8 and a["row_id"].dtype == np.int32 and a["problem_id"].dtype == np.int32
        assert a["sample_idx"].dtype == np.int8
    assert not set(tr["row_id"]) & set(te["row_id"])
    assert all(np.all(np.diff(a["row_id"]) > 0) for a in (ho, se))
    assert ho["row_id"][600] == 4000 and se["row_id"][600] == 4000 + 450 * 4   # extra rows follow the original ones
    # companion train holds all public rows but in shuffled order, so ungrouped CV leaks
    assert sorted(ct["row_id"].tolist()) == sorted(set(tr["row_id"].tolist()) | set(te["row_id"].tolist()))
    assert not np.all(np.diff(ct["row_id"]) > 0)


def test_without_extra_keeps_the_original_allocation(runs):
    f = _load(runs["base"])
    assert (len(f["holdout"]["y"]), len(f["secret"]["y"])) == (600, 600)
    assert set(f["holdout"]["problem_id"]) == set(range(700, 850))
    assert set(f["secret"]["problem_id"]) == set(range(850, 1000))


def test_public_files_are_byte_identical_with_and_without_extra(runs):
    for name in PUBLIC:
        assert _sha(runs["base"] / PATHS[name]) == _sha(runs["ext"] / PATHS[name]), name
    assert _sha(runs["base"] / PATHS["holdout"]) != _sha(runs["ext"] / PATHS["holdout"])


def test_stats_file_gains_extra_combined_and_splits(runs):
    s = json.loads(runs["stats"].read_text())
    assert list(s) == ["original", "extra", "combined", "splits"]
    assert s["original"]["n_rows"] == 4000 and s["original"]["marker_counts"] == {"hashes": 1}
    assert s["extra"]["n_rows"] == 4800 and s["extra"]["problem_id_range"] == [1000, 2199]
    assert s["combined"]["rows"] == 8800 and s["combined"]["problems"] == 2200
    assert s["combined"]["problem_id_runs"] == [[0, 2199]]
    sizes = {k: (v["rows"], v["problems"]) for k, v in s["splits"].items()}
    assert sizes == {"train": (2100, len(set(np.load(runs["ext"] / PATHS["train"])["problem_id"]))),
                     "test": (700, len(set(np.load(runs["ext"] / PATHS["test"])["problem_id"]))),
                     "holdout": (2400, 600), "companion_train": (2800, 700), "secret": (3600, 900)}
    assert s["splits"]["holdout"]["problem_id_runs"] == [[700, 849], [1000, 1449]]
    assert s["splits"]["secret"]["problem_id_runs"] == [[850, 999], [1450, 2199]]
    assert s["splits"]["holdout"]["problem_id_range"] == [700, 1449] and s["splits"]["secret"]["problem_id_range"] == [850, 2199]
    assert s["splits"]["companion_train"]["problem_id_runs"] == [[0, 699]]
    assert "problem_id_runs" not in s["splits"]["train"] and "problem_id_runs" not in s["splits"]["test"]
    assert s["splits"]["train"]["problem_id_range"] == [0, 699]
    assert s["splits"]["holdout"]["path"] == PATHS["holdout"]
    for v in s["splits"].values():
        assert 0.0 <= v["positive_rate"] <= 1.0 and v["positives"] <= v["rows"]
    # a second update of the already restructured file is idempotent in shape
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(runs["stats"].parent / "activations_full.npz"),
                    "--extra", str(runs["stats"].parent / "activations_extra.npz"),
                    "--repo", str(runs["ext"]), "--stats", str(runs["stats"])], check=True)
    assert json.loads(runs["stats"].read_text()) == s


def test_allocate_rejects_overlapping_or_unranged_extra():
    full = _synthetic(1000, 0, 0, seed=0)
    with pytest.raises(AssertionError, match="problem_id overlap"):
        allocate(full, _synthetic(10, 995, 4000, seed=2))
    with pytest.raises(AssertionError, match="row_id overlap"):
        allocate(full, _synthetic(10, 1000, 3990, seed=2))
    with pytest.raises(AssertionError, match="outside every range"):
        allocate(full, _synthetic(1201, 1000, 4000, seed=2))


def test_runs_helper():
    assert _runs([]) == []
    assert _runs([700, 701, 702]) == [[700, 702]]
    assert _runs(np.array([700, 701, 1000, 1001, 1002, 1450])) == [[700, 701], [1000, 1002], [1450, 1450]]
