"""Fast unit tests for tools/calibrate.py.

The key-presence test runs ``compute_calibration`` on a tiny synthetic repo
fixture (a handful of problems, 8 feature columns) so it stays fast; it never
touches the real 896-dimensional data. The BAR and TOL tests exercise the pure
formula/clamp helpers with hand-made inputs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

# tools/ holds calibrate.py; this test lives in tools/tests/.
TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import calibrate  # noqa: E402

EXPECTED_KEYS = {
    "honest_cv_prompt",
    "honest_cv_reason",
    "honest_cv_concat",
    "leaky_row_cv_prompt",
    "holdout_auroc_per_legit",
    "companion_secret_auroc",
    "companion_grouped_estimate",
    "companion_estimate_gap_sd",
    "companion_leaky_estimate",
    "BAR",
    "TOL",
}


# ------------------------------------------------------------- synthetic data

def _make_split(rng, n_problems, start_pid, d=8):
    """One npz-worth of rows: 4 samples per problem, a signal in X_reason[:, 0].

    X_prompt is shared across a problem's four rows (as in the real data), so a
    row-wise split leaks it while a grouped split does not.
    """
    rows = n_problems * 4
    pid = np.repeat(np.arange(start_pid, start_pid + n_problems), 4).astype(np.int32)
    y = rng.integers(0, 2, size=rows).astype(np.int8)
    x_reason = rng.normal(0, 1, (rows, d)).astype(np.float64)
    x_reason[:, 0] += y * 1.5  # a real, row-level signal
    x_prompt = np.zeros((rows, d), dtype=np.float64)
    for q in np.unique(pid):
        mask = pid == q
        # a per-problem constant that leans on the problem's mean label
        x_prompt[mask] = rng.normal(0, 1, d) + y[mask].mean()
    return {
        "X_prompt": x_prompt.astype(np.float16),
        "X_reason": x_reason.astype(np.float16),
        "y": y,
        "problem_id": pid,
        "sample_idx": (np.arange(rows) % 4).astype(np.int8),
        "row_id": (start_pid * 4 + np.arange(rows)).astype(np.int32),
    }


def _build_repo(tmp_path):
    rng = np.random.default_rng(0)
    flagship = tmp_path / "grader-hardening/environment/app/env/public/data"
    flagship.mkdir(parents=True)
    np.savez(flagship / "train.npz", **_make_split(rng, 12, 0))

    out = tmp_path / "tools/output"
    out.mkdir(parents=True)
    pool = {name: {"holdout_auroc": val, "cv_auroc": val - 0.02}
            for name, val in zip(
                ("L1", "L2", "L3", "L4", "L5", "L6"),
                (0.66, 0.83, 0.82, 0.82, 0.68, 0.66))}
    pool["t01"] = {"holdout_auroc": 0.64, "cv_auroc": 0.61}  # a non-legit entry
    (out / "pool_report.json").write_text(json.dumps(pool))

    comp = tmp_path / "honest-probe/environment/app/data"
    comp.mkdir(parents=True)
    np.savez(comp / "train.npz", **_make_split(rng, 16, 0))
    secret = tmp_path / "honest-probe/tests/private"
    secret.mkdir(parents=True)
    np.savez(secret / "test.npz", **_make_split(rng, 10, 500))
    return tmp_path


def test_calibrate_produces_all_keys(tmp_path):
    repo = _build_repo(tmp_path)
    result = calibrate.compute_calibration(repo)
    assert set(result) == EXPECTED_KEYS
    # holdout is copied straight from the pool report, legit entries only
    assert set(result["holdout_auroc_per_legit"]) == {"L1", "L2", "L3", "L4", "L5", "L6"}
    # every AUROC-like quantity is a plain float in [0, 1]
    for key in ("honest_cv_prompt", "honest_cv_reason", "honest_cv_concat",
                "leaky_row_cv_prompt", "companion_secret_auroc",
                "companion_grouped_estimate", "companion_leaky_estimate"):
        assert isinstance(result[key], float)
        assert 0.0 <= result[key] <= 1.0
    assert result["BAR"] == round(result["BAR"], 2)
    assert result["TOL"] == round(result["TOL"], 2)


# ----------------------------------------------------------- rounding helpers

@pytest.mark.parametrize("x,expected", [
    (0.6201283, 0.62),
    (0.629, 0.62),
    (0.55, 0.55),
    (0.70, 0.70),
    (0.549999, 0.54),
])
def test_floor_to_01(x, expected):
    assert calibrate.floor_to_01(x) == expected


@pytest.mark.parametrize("x,expected", [
    (0.0, 0.0),
    (0.04, 0.04),
    (0.041, 0.05),
    (0.062, 0.07),
    (0.052, 0.06),
])
def test_ceil_to_01(x, expected):
    assert calibrate.ceil_to_01(x) == expected


# ---------------------------------------------------------------- BAR formula

@pytest.mark.parametrize("holdout,expected", [
    ({"L1": 0.6601283, "L2": 0.83}, 0.62),            # floor(0.6201) = 0.62
    ({"L1": 0.70, "L2": 0.66}, 0.62),                 # min is 0.66
    ({"L1": 0.58, "L2": 0.90}, 0.55),                 # floor(0.54) -> clamp up to 0.55
    ({"L1": 0.50, "L2": 0.90}, 0.55),                 # clamp up to 0.55
    ({"L1": 0.80, "L2": 0.95}, 0.70),                 # floor(0.76) -> clamp down to 0.70
    ({"L1": 0.59, "L2": 0.95}, 0.55),                 # floor(0.55) = 0.55, no clamp
])
def test_bar_from_holdout(holdout, expected):
    assert calibrate.bar_from_holdout(holdout) == expected


# ---------------------------------------------------------------- TOL formula

@pytest.mark.parametrize("gap_sd,expected", [
    (0.0, 0.05),       # floor at 0.05
    (0.02, 0.05),      # 2*sd = 0.04, still below floor
    (0.025, 0.05),     # 2*sd = 0.05 -> max(0.05, 0.05)
    (0.026, 0.06),     # 2*sd = 0.052 -> ceil 0.06
    (0.03, 0.06),      # 2*sd = 0.06
    (0.031, 0.07),     # 2*sd = 0.062 -> ceil 0.07
])
def test_tol_from_gap_sd(gap_sd, expected):
    assert calibrate.tol_from_gap_sd(gap_sd) == expected


# ------------------------------------------------------------------ the gates

def test_check_loud_failures_flags_small_leakage():
    result = {
        "BAR": 0.62,
        "honest_cv_prompt": 0.60,
        "leaky_row_cv_prompt": 0.65,  # gap 0.05 < 0.10
        "holdout_auroc_per_legit": {"L1": 0.70},
        "companion_secret_auroc": 0.70,
    }
    failures = calibrate.check_loud_failures(result)
    assert any("leakage too small" in f for f in failures)


def test_check_loud_failures_clean_passes():
    result = {
        "BAR": 0.62,
        "honest_cv_prompt": 0.60,
        "leaky_row_cv_prompt": 0.75,  # gap 0.15 >= 0.10
        "holdout_auroc_per_legit": {"L1": 0.66, "L2": 0.83},
        "companion_secret_auroc": 0.70,
    }
    assert calibrate.check_loud_failures(result) == []


def test_check_trap_robustness_clean_passes():
    result = {
        "TOL": 0.05,
        "companion_secret_auroc": 0.70,
        "companion_leaky_estimate": 0.80,   # |0.80-0.70| = 0.10 > 0.06
        "companion_grouped_estimate": 0.71,  # |0.71-0.70| = 0.01
        "companion_estimate_gap_sd": 0.01,   # 0.01 + 0.01 = 0.02 < 0.05
    }
    assert calibrate.check_trap_robustness(result) == []


def test_check_trap_robustness_flags_overlap():
    result = {
        "TOL": 0.05,
        "companion_secret_auroc": 0.70,
        "companion_leaky_estimate": 0.72,   # |0.72-0.70| = 0.02, not > 0.06
        "companion_grouped_estimate": 0.76,  # |0.76-0.70| = 0.06, not < 0.05
        "companion_estimate_gap_sd": 0.02,
    }
    concerns = calibrate.check_trap_robustness(result)
    assert len(concerns) == 2
