"""Fast unit tests for tools/calibrate.py.

The key-presence test runs ``compute_calibration`` on a tiny synthetic repo
fixture (a handful of problems, 8 feature columns) so it stays fast; it never
touches the real 896-dimensional data. The BAR, WIDTH, interval and TOL tests
exercise the pure formula/clamp helpers with hand-made inputs. The cluster
bootstrap is tested on synthetic clustered scores: constant within a problem,
as an X_prompt probe's scores are, so the problem-level sd must exceed the
row-level one.
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
    "flagship_bar_inputs",
    "companion_secret_auroc",
    "companion_grouped_estimate",
    "companion_estimate_gap_sd",
    "companion_leaky_estimate",
    "companion_leaky_gap",
    "companion_secret_bootstrap",
    "companion_random_scores",
    "companion_canonical_folds",
    "companion_canonical_interval",
    "sd_m",
    "sd_e",
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
                ("L1", "L2", "L3", "L4", "L5", "L6", "L7"),
                (0.6334, 0.83, 0.82, 0.82, 0.68, 0.6335, 0.6524))}
    pool["t01"] = {"holdout_auroc": 0.64, "cv_auroc": 0.61}  # a non-legit entry
    pool["t11_weak_probe"] = {"rows": 40, "holdout_auroc": 0.5613}
    pool["holdout_rows"] = 2400
    pool["interpreter_flags"] = ["-E"]
    (out / "pool_report.json").write_text(json.dumps(pool))

    comp = tmp_path / "honest-probe/environment/app/data"
    comp.mkdir(parents=True)
    np.savez(comp / "train.npz", **_make_split(rng, 40, 0))
    secret = tmp_path / "honest-probe/tests/private"
    secret.mkdir(parents=True)
    np.savez(secret / "test.npz", **_make_split(rng, 10, 500))
    return tmp_path


def test_calibrate_produces_all_keys(tmp_path):
    repo = _build_repo(tmp_path)
    result = calibrate.compute_calibration(repo)
    assert set(result) == EXPECTED_KEYS
    # holdout is copied straight from the pool report, legit entries only
    assert set(result["holdout_auroc_per_legit"]) == {"L1", "L2", "L3", "L4", "L5", "L6", "L7"}
    # the flagship bar is the rounded midpoint of the weakest honest item and the weak probe
    assert result["BAR"] == 0.60
    inputs = result["flagship_bar_inputs"]
    assert inputs["min_honest_id"] == "L1"
    assert inputs["min_honest_holdout_auroc"] == 0.6334
    assert inputs["weak_probe_holdout_auroc"] == 0.5613
    assert inputs["midpoint"] == pytest.approx(0.59735)
    assert inputs["margin_honest"] == pytest.approx(0.0334)
    assert inputs["margin_weak"] == pytest.approx(0.0387)
    assert inputs["holdout_rows"] == 2400 and inputs["holdout_problems"] == 600
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
    entry = result["companion_random_scores"]
    assert 0.0 <= entry["auroc"] <= 1.0
    assert entry["sd"] >= 0.0
    assert isinstance(entry["below_bar_by_3sd"], bool)
    assert entry["seed"] == 20261107
    folds = result["companion_canonical_folds"]
    assert folds["n_splits"] == 10 and folds["seeds"] == [0, 1, 2, 3, 4]
    assert folds["n_folds"] + folds["n_skipped"] == 50
    assert folds["n_skipped"] == 0
    assert folds["train_problems"] == 40
    assert folds["problems_per_fold"] == pytest.approx(4.0)
    assert 0.0 <= folds["mean"] <= 1.0 and folds["sd"] >= 0.0
    assert folds["sd_e"] == pytest.approx(folds["sd"] / 10 ** 0.5)
    assert result["sd_e"] == folds["sd_e"] and result["sd_m"] == boot["sd"]
    canon = result["companion_canonical_interval"]
    assert canon["secret_problems"] == 10
    assert canon["half_width"] == pytest.approx(calibrate.interval_half_width(
        folds["sd"], folds["problems_per_fold"], 10, 10))
    assert canon["lo"] == pytest.approx(folds["mean"] - canon["half_width"])
    assert canon["hi"] == pytest.approx(folds["mean"] + canon["half_width"])
    assert canon["width"] == pytest.approx(2 * canon["half_width"])
    assert canon["contains"] is (canon["lo"] <= result["companion_secret_auroc"] <= canon["hi"])
    assert canon["margin_lo"] == pytest.approx(result["companion_secret_auroc"] - canon["lo"])
    assert canon["margin_hi"] == pytest.approx(canon["hi"] - result["companion_secret_auroc"])
    assert result["WIDTH"] == calibrate.width_from_sds(result["sd_m"], result["sd_e"])
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


def test_canonical_fold_aurocs_follow_the_canonical_scheme():
    rng = np.random.default_rng(0)
    tr = _make_split(rng, 20, 0)
    X = tr["X_prompt"].astype(np.float64)
    y = tr["y"].astype(np.int64)
    g = tr["problem_id"].astype(np.int64)
    folds, ppf, skipped = calibrate.canonical_fold_aurocs(X, y, g, auroc, n_splits=5, seeds=(0, 1))
    assert len(folds) + skipped == 10
    assert skipped == 0
    assert ppf == pytest.approx(4.0)  # 20 problems in 5 folds
    assert all(0.0 <= v <= 1.0 for v in folds)
    again, _, _ = calibrate.canonical_fold_aurocs(X, y, g, auroc, n_splits=5, seeds=(0, 1))
    assert again == folds  # seeded


def test_canonical_fold_aurocs_skips_single_class_folds():
    # Two problems, one all-positive and one all-negative: every 2-fold
    # validation fold holds one class, so nothing is usable.
    X = np.random.default_rng(0).normal(size=(8, 3))
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    with pytest.raises(ValueError):
        calibrate.canonical_fold_aurocs(X, y, g, auroc, n_splits=2, seeds=(0,))


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

@pytest.mark.parametrize("x,expected", [
    (0.59735, 0.60),
    (0.605, 0.61),     # half up, not banker's rounding
    (0.604999, 0.60),
    (0.55, 0.55),
])
def test_round_to_01(x, expected):
    assert calibrate.round_to_01(x) == expected


@pytest.mark.parametrize("holdout,weak,expected,worst", [
    ({"L1": 0.6334, "L2": 0.83, "L6": 0.6335}, 0.5613, 0.60, "L1"),  # the current rebuild
    ({"L1": 0.70, "L2": 0.66}, 0.50, 0.58, "L2"),                     # midpoint 0.58
    ({"L1": 0.61, "L2": 0.90}, 0.60, 0.61, "L1"),                     # 0.605 rounds half up
    ({"L1": 0.58, "L2": 0.90}, 0.50, 0.55, "L1"),                     # 0.54 -> clamp up to 0.55
    ({"L1": 0.95, "L2": 0.96}, 0.60, 0.70, "L1"),                     # 0.775 -> clamp down to 0.70
])
def test_bar_from_midpoint(holdout, weak, expected, worst):
    bar, inputs = calibrate.bar_from_midpoint(holdout, weak)
    assert bar == expected
    assert inputs["min_honest_id"] == worst
    assert inputs["midpoint"] == pytest.approx((holdout[worst] + weak) / 2)
    assert inputs["margin_honest"] == pytest.approx(holdout[worst] - bar)
    assert inputs["margin_weak"] == pytest.approx(bar - weak)


def test_bar_from_midpoint_rejects_bad_input():
    with pytest.raises(ValueError):
        calibrate.bar_from_midpoint({}, 0.56)
    with pytest.raises(ValueError):
        calibrate.bar_from_midpoint({"L1": 0.63}, float("nan"))


def test_compute_calibration_requires_the_weak_probe_entry(tmp_path):
    repo = _build_repo(tmp_path)
    pool_path = repo / "tools/output/pool_report.json"
    pool = json.loads(pool_path.read_text())
    del pool["t11_weak_probe"]
    pool_path.write_text(json.dumps(pool))
    with pytest.raises(ValueError, match="t11_weak_probe"):
        calibrate.compute_calibration(repo)


# -------------------------------------------------------------- WIDTH formula

@pytest.mark.parametrize("sd_m,sd_e,expected", [
    (0.0, 0.0, 0.06),          # clamp up to 0.06
    (0.01, 0.0, 0.06),         # 4*0.01 = 0.04 -> clamp up
    (0.015, 0.0, 0.06),        # 4*0.015 = 0.06 exactly
    (0.0151, 0.0, 0.07),       # 0.0604 -> ceil 0.07
    (0.0131, 0.013, 0.08),     # 4*sqrt(0.0131^2+0.013^2) = 0.0738 -> 0.08 (the 900-problem case)
    (0.0131, 0.0058, 0.06),    # 0.0573 -> 0.06 (the literal n_folds=50 reading)
    (0.0303, 0.0, 0.12),       # 0.1212 -> clamp down (the 150-problem case)
    (0.03, 0.03, 0.12),        # 0.1697 -> clamp down
    (0.0, 0.02, 0.08),         # sd_e alone: 0.08 exactly
])
def test_width_from_sds(sd_m, sd_e, expected):
    assert calibrate.width_from_sds(sd_m, sd_e) == expected


def test_width_from_sds_rejects_bad_sd():
    with pytest.raises(ValueError):
        calibrate.width_from_sds(float("nan"), 0.01)
    with pytest.raises(ValueError):
        calibrate.width_from_sds(0.01, -0.01)


@pytest.mark.parametrize("sd,ppf,k,secret,expected", [
    (0.0411, 70, 10, 900, 2 * (0.0411 ** 2 * 70 / 900 + 0.0411 ** 2 / 10) ** 0.5),  # canonical at 900
    (0.0411, 70, 10, 150, 2 * (0.0411 ** 2 * 70 / 150 + 0.0411 ** 2 / 10) ** 0.5),  # canonical at 150
    (0.02, 70, 10, 900, 2 * 0.02 * (70 / 900 + 0.1) ** 0.5),
    (0.0, 70, 10, 900, 0.0),
])
def test_interval_half_width(sd, ppf, k, secret, expected):
    assert calibrate.interval_half_width(sd, ppf, k, secret) == pytest.approx(expected)


def test_interval_half_width_repeats_do_not_shrink_the_estimate_term():
    # Same sd and fold geometry: the half-width must not depend on how many
    # times the partition was repeated (which is what a raw n_folds would do).
    a = calibrate.interval_half_width(0.04, 70, 10, 900)
    assert a == pytest.approx(2 * (0.04 ** 2 * 70 / 900 + 0.04 ** 2 / 10) ** 0.5)
    assert a > 2 * 0.04 * (70 / 900) ** 0.5  # strictly wider than the measurement term alone


def test_interval_half_width_rejects_bad_input():
    with pytest.raises(ValueError):
        calibrate.interval_half_width(float("nan"), 70, 10, 900)
    with pytest.raises(ValueError):
        calibrate.interval_half_width(0.04, 0, 10, 900)
    with pytest.raises(ValueError):
        calibrate.interval_half_width(0.04, 70, 10, 0)


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
        "BAR": 0.60,
        "honest_cv_prompt": 0.60,
        "leaky_row_cv_prompt": 0.65,  # gap 0.05 < 0.10
        "holdout_auroc_per_legit": {"L1": 0.70},
        "companion_secret_auroc": 0.70,
    }
    failures = calibrate.check_loud_failures(result)
    assert any("leakage too small" in f for f in failures)


def test_check_loud_failures_clean_passes():
    result = {
        "BAR": 0.60,
        "honest_cv_prompt": 0.60,
        "leaky_row_cv_prompt": 0.75,  # gap 0.15 >= 0.10
        "holdout_auroc_per_legit": {"L1": 0.6334, "L2": 0.83},
        "flagship_bar_inputs": {"weak_probe_holdout_auroc": 0.5613},
        "companion_secret_auroc": 0.70,
        "companion_secret_bootstrap": {"sd": 0.03, "clears_bar_by_3sd": True},
    }
    assert calibrate.check_loud_failures(result) == []


def test_check_loud_failures_flags_bar_not_separating():
    result = {
        "BAR": 0.60,
        "honest_cv_prompt": 0.60,
        "leaky_row_cv_prompt": 0.75,
        "holdout_auroc_per_legit": {"L1": 0.60, "L2": 0.83},   # not strictly above
        "flagship_bar_inputs": {"weak_probe_holdout_auroc": 0.61},  # above the bar
        "companion_secret_auroc": 0.70,
        "companion_secret_bootstrap": {"sd": 0.03, "clears_bar_by_3sd": True},
    }
    failures = calibrate.check_loud_failures(result)
    assert any("legit L1" in f for f in failures)
    assert any("weak probe" in f for f in failures)


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


def test_check_trap_robustness_flags_thin_flagship_margin():
    result = {
        "WIDTH": 0.08,
        "flagship_bar_inputs": {"margin_honest": 0.012, "margin_weak": 0.05},
        "companion_secret_auroc": 0.66,
        "companion_leaky_estimate": 0.78,
        "companion_grouped_estimate": 0.65,
        "companion_random_scores": {"auroc": 0.49, "sd": 0.01, "below_bar_by_3sd": True},
        "companion_canonical_interval": {"lo": 0.62, "hi": 0.69, "width": 0.07, "contains": True},
    }
    concerns = calibrate.check_trap_robustness(result)
    assert len(concerns) == 1 and "margin_honest" in concerns[0]


def test_check_trap_robustness_clean_passes():
    result = {
        "WIDTH": 0.08,
        "flagship_bar_inputs": {"margin_honest": 0.0334, "margin_weak": 0.0387},
        "companion_secret_auroc": 0.66,
        "companion_leaky_estimate": 0.78,   # gap 0.12 > 0.08
        "companion_grouped_estimate": 0.65,  # gap 0.01 < 0.04
        "companion_random_scores": {"auroc": 0.49, "sd": 0.01, "below_bar_by_3sd": True},
        "companion_canonical_interval": {"lo": 0.62, "hi": 0.69, "width": 0.07, "contains": True},
    }
    assert calibrate.check_trap_robustness(result) == []


def test_check_trap_robustness_flags_every_overlap():
    result = {
        "WIDTH": 0.06,
        "companion_secret_auroc": 0.66,
        "companion_leaky_estimate": 0.71,   # gap 0.05, not > 0.06
        "companion_grouped_estimate": 0.62,  # gap 0.04, not < 0.03
        "companion_random_scores": {"auroc": 0.49, "sd": 0.03, "below_bar_by_3sd": False},
        "companion_canonical_interval": {"lo": 0.60, "hi": 0.65, "width": 0.07, "contains": False},
    }
    concerns = calibrate.check_trap_robustness(result)
    assert len(concerns) == 5
    assert any("naive estimate" in c for c in concerns)
    assert any("honest estimate" in c for c in concerns)
    assert any("random scores" in c for c in concerns)
    assert any("does not contain" in c for c in concerns)
    assert any("more than WIDTH" in c for c in concerns)
    assert not any("weak probe" in c for c in concerns)
