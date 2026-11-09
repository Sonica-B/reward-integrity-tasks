"""Fast unit tests for tools/calibrate.py.

The key-presence test runs ``compute_calibration`` on a tiny synthetic repo
fixture (a handful of problems, 8 feature columns) so it stays fast; it never
touches the real 896-dimensional data. The BAR, WIDTH and TOL tests exercise
the pure formula/clamp helpers with hand-made inputs. The cluster bootstrap is
tested on synthetic clustered scores: constant within a problem, as an
X_prompt probe's scores are, so the problem-level sd must exceed the row-level
one.
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
sys.path.insert(0, str(TOOLS / "common"))

import calibrate  # noqa: E402
from auroc import auroc  # noqa: E402

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
    "companion_leaky_gap",
    "companion_secret_bootstrap",
    "companion_random_scores",
    "companion_weak_probe",
    "BAR",
    "BAR_companion",
    "WIDTH",
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


def _clustered_scores(rng, n_problems, rows_per=4, strength=1.0):
    """Labels and scores where the score is constant within a problem.

    Each problem has a latent difficulty; its rows' labels are Bernoulli in that
    difficulty and its score is the difficulty plus noise, so the AUROC is well
    above chance and the rows of a problem are strongly dependent.
    """
    latent = rng.normal(0, 1, n_problems)
    p = 1.0 / (1.0 + np.exp(-strength * latent))
    y = (rng.random((n_problems, rows_per)) < p[:, None]).astype(np.int64).reshape(-1)
    scores = np.repeat(latent + rng.normal(0, 0.5, n_problems), rows_per)
    groups = np.repeat(np.arange(n_problems), rows_per)
    return y, scores, groups


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
    assert result["BAR_companion"] == 0.55
    assert result["WIDTH"] == round(result["WIDTH"], 2)
    assert 0.06 <= result["WIDTH"] <= 0.12
    boot = result["companion_secret_bootstrap"]
    assert boot["n_boot"] == 2000 and boot["seed"] == calibrate.BOOT_SEED
    assert boot["n_problems"] == 10 and boot["n_rows"] == 40
    assert boot["p2_5"] <= boot["p97_5"]
    assert boot["sd"] >= 0.0
    assert isinstance(boot["clears_bar_by_3sd"], bool)
    for key in ("companion_random_scores", "companion_weak_probe"):
        entry = result[key]
        assert 0.0 <= entry["auroc"] <= 1.0
        assert entry["sd"] >= 0.0
        assert isinstance(entry["below_bar_by_3sd"], bool)
    assert result["companion_weak_probe"]["train_rows"] == 40
    assert len(result["companion_weak_probe"]["train_problems"]) == 10
    by_seed = result["companion_weak_probe"]["auroc_by_seed"]
    assert set(by_seed) == {"0", "1", "2", "3", "4"}
    assert by_seed["0"] == pytest.approx(result["companion_weak_probe"]["auroc"])
    assert all(0.0 <= v <= 1.0 for v in by_seed.values())
    assert result["companion_random_scores"]["seed"] == 20261107
    assert result["companion_leaky_gap"] == pytest.approx(
        result["companion_leaky_estimate"] - result["companion_secret_auroc"])
    # the result must be JSON-serialisable as written
    json.dumps(result)


# ---------------------------------------------------------- cluster bootstrap

def test_cluster_bootstrap_is_deterministic_and_well_formed():
    y, s, g = _clustered_scores(np.random.default_rng(1), 150)
    a = calibrate.cluster_bootstrap_auroc(y, s, g, auroc, n_boot=300, seed=7)
    b = calibrate.cluster_bootstrap_auroc(y, s, g, auroc, n_boot=300, seed=7)
    assert a == b
    assert a["point"] == pytest.approx(auroc(y, s))
    assert a["n_clusters"] == 150 and a["n_rows"] == 600
    assert a["n_valid"] + a["n_degenerate"] == 300 == a["n_boot"]
    assert a["n_degenerate"] == 0
    assert a["p2_5"] <= a["point"] <= a["p97_5"]
    assert 0.0 < a["sd"] < 0.1
    c = calibrate.cluster_bootstrap_auroc(y, s, g, auroc, n_boot=300, seed=8)
    assert c["sd"] != a["sd"]  # a different seed draws different resamples


def test_cluster_bootstrap_exceeds_row_bootstrap_on_clustered_scores():
    """Scores constant within a problem: the effective n is the problems, so the
    problem-level sd must be clearly larger than a row-level resample's."""
    y, s, g = _clustered_scores(np.random.default_rng(2), 150)
    cluster = calibrate.cluster_bootstrap_auroc(y, s, g, auroc, n_boot=2000, seed=3)
    rows = calibrate.cluster_bootstrap_auroc(y, s, np.arange(y.size), auroc, n_boot=2000, seed=3)
    assert cluster["sd"] > 1.15 * rows["sd"], (cluster["sd"], rows["sd"])


def test_cluster_bootstrap_matches_row_bootstrap_when_rows_are_independent():
    """With one row per cluster both procedures resample the same units."""
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 600)
    s = rng.normal(0, 1, 600) + 0.8 * y
    a = calibrate.cluster_bootstrap_auroc(y, s, np.arange(600), auroc, n_boot=2000, seed=5)
    b = calibrate.cluster_bootstrap_auroc(y, s, np.arange(600) + 1000, auroc, n_boot=2000, seed=5)
    assert a["sd"] == pytest.approx(b["sd"])
    assert 0.01 < a["sd"] < 0.04  # Hanley-McNeil scale for n = 600 at AUROC near 0.7


def test_cluster_bootstrap_perfect_separation_has_zero_spread():
    y = np.repeat([0, 1], 20)
    s = np.repeat([0.0, 1.0], 20)
    g = np.arange(40) // 4
    out = calibrate.cluster_bootstrap_auroc(y, s, g, auroc, n_boot=200, seed=0)
    assert out["point"] == 1.0
    assert out["sd"] == 0.0
    assert out["p2_5"] == out["p97_5"] == 1.0


def test_cluster_bootstrap_counts_degenerate_resamples():
    # Three clusters, one of them holding every positive: a resample that
    # misses it has one class and must be skipped, not scored.
    y = np.array([1, 1, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1, 0.3, 0.4])
    g = np.array([0, 0, 1, 1, 2, 2])
    out = calibrate.cluster_bootstrap_auroc(y, s, g, auroc, n_boot=400, seed=0)
    assert out["n_degenerate"] > 0
    assert out["n_valid"] + out["n_degenerate"] == 400
    assert out["point"] == 1.0


def test_cluster_bootstrap_rejects_bad_input():
    with pytest.raises(ValueError):
        calibrate.cluster_bootstrap_auroc([0, 1], [0.1], [0, 1], auroc)
    with pytest.raises(ValueError):
        calibrate.cluster_bootstrap_auroc([], [], [], auroc)
    with pytest.raises(ValueError):
        calibrate.cluster_bootstrap_auroc([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1], auroc, n_boot=1)
    with pytest.raises(ValueError):
        # every resample degenerate: a single cluster holding one class only
        calibrate.cluster_bootstrap_auroc([0, 0, 0], [0.1, 0.2, 0.3], [0, 0, 0], auroc, n_boot=10)


def test_weak_probe_uses_the_requested_problem_count():
    rng = np.random.default_rng(0)
    tr = _make_split(rng, 16, 0)
    te = _make_split(rng, 10, 500)
    scores, problems, rows = calibrate.weak_probe_scores(
        tr["X_prompt"].astype(np.float64), tr["y"].astype(np.int64),
        tr["problem_id"].astype(np.int64), te["X_prompt"].astype(np.float64))
    assert scores.shape == (40,)
    assert len(problems) == 10 and rows == 40
    assert len(set(problems)) == 10


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


# -------------------------------------------------------------- WIDTH formula

@pytest.mark.parametrize("sd,expected", [
    (0.0, 0.06),        # clamp up to 0.06
    (0.01, 0.06),       # 4*sd = 0.04 -> clamp up
    (0.0127, 0.06),     # 4*sd = 0.0508 -> ceil 0.06
    (0.015, 0.06),      # 4*sd = 0.06 exactly
    (0.0151, 0.07),     # 4*sd = 0.0604 -> ceil 0.07
    (0.02, 0.08),       # 4*sd = 0.08 exactly
    (0.0201, 0.09),     # ceil
    (0.031, 0.12),      # 4*sd = 0.124 -> clamp down to 0.12
    (0.05, 0.12),       # clamp down
])
def test_width_from_sd(sd, expected):
    assert calibrate.width_from_sd(sd) == expected


def test_width_from_sd_rejects_bad_sd():
    with pytest.raises(ValueError):
        calibrate.width_from_sd(float("nan"))
    with pytest.raises(ValueError):
        calibrate.width_from_sd(-0.01)


@pytest.mark.parametrize("bar,point,sd,expected", [
    (0.55, 0.48, 0.02, True),     # 0.07 >= 0.06
    (0.55, 0.50, 0.02, False),    # 0.05 < 0.06
    (0.55, 0.49, 0.02, True),     # 0.06 >= 0.06 (boundary)
    (0.55, 0.60, 0.01, False),    # above the bar
])
def test_below_bar_by_sd(bar, point, sd, expected):
    assert calibrate.below_bar_by_sd(bar, point, sd) is expected


@pytest.mark.parametrize("bar,point,sd,expected", [
    (0.55, 0.663, 0.031, True),   # 0.113 >= 0.093
    (0.55, 0.60, 0.031, False),   # 0.05 < 0.093
    (0.55, 0.60, 0.0127, True),   # 0.05 >= 0.0381
])
def test_above_bar_by_sd(bar, point, sd, expected):
    assert calibrate.above_bar_by_sd(bar, point, sd) is expected


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
        "companion_secret_bootstrap": {"sd": 0.03, "clears_bar_by_3sd": True},
    }
    assert calibrate.check_loud_failures(result) == []


def test_check_loud_failures_flags_companion_too_close_to_its_bar():
    result = {
        "BAR": 0.62,
        "honest_cv_prompt": 0.60,
        "leaky_row_cv_prompt": 0.75,
        "holdout_auroc_per_legit": {"L1": 0.66},
        "companion_secret_auroc": 0.60,  # 0.05 above 0.55, sd 0.03 needs 0.09
        "companion_secret_bootstrap": {"sd": 0.03, "clears_bar_by_3sd": False},
    }
    failures = calibrate.check_loud_failures(result)
    assert any("BAR_companion" in f for f in failures)


def test_check_trap_robustness_clean_passes():
    result = {
        "WIDTH": 0.06,
        "companion_secret_auroc": 0.66,
        "companion_leaky_estimate": 0.78,   # gap 0.12 > 0.06
        "companion_grouped_estimate": 0.65,  # gap 0.01 < 0.03
        "companion_random_scores": {"auroc": 0.49, "sd": 0.01, "below_bar_by_3sd": True},
        "companion_weak_probe": {"auroc": 0.51, "sd": 0.01, "below_bar_by_3sd": True},
    }
    assert calibrate.check_trap_robustness(result) == []


def test_check_trap_robustness_flags_every_overlap():
    result = {
        "WIDTH": 0.12,
        "companion_secret_auroc": 0.66,
        "companion_leaky_estimate": 0.75,   # gap 0.09, not > 0.12
        "companion_grouped_estimate": 0.59,  # gap 0.07, not < 0.06
        "companion_random_scores": {"auroc": 0.49, "sd": 0.03, "below_bar_by_3sd": False},
        "companion_weak_probe": {"auroc": 0.53, "sd": 0.03, "below_bar_by_3sd": False},
    }
    concerns = calibrate.check_trap_robustness(result)
    assert len(concerns) == 4
    assert any("naive estimate" in c for c in concerns)
    assert any("honest estimate" in c for c in concerns)
    assert any("random scores" in c for c in concerns)
    assert any("weak probe" in c for c in concerns)
