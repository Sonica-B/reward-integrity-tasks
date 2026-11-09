"""Calibrate the verifier constants from the data.

Run as::

    python tools/calibrate.py --repo .

The script measures, on the committed data, the quantities that set the
constants used by the Harbor tasks:

* BAR  (flagship) the AUROC a probe must clear in grader-hardening: the
  midpoint between the weakest honest pool solution's holdout AUROC (L1 to
  L7) and the t11 weak probe's holdout AUROC, rounded half up to 0.01 and
  clamped to [0.55, 0.70]. Both inputs come from tools/output/pool_report.json
  and are recorded with the margins on each side and the holdout size.
* BAR_companion  fixed at 0.55 for honest-probe: a floor well above chance
  that is decoupled from where the reference probe lands. Recorded with it is
  the check that random scores stay below it by at least 3 standard
  deviations of their own measurement, and the loud check that the canonical
  probe clears it by at least 3 standard deviations. (A probe fit on 10
  problems reaches 0.59 to 0.69 on this data, so "weak probe below the bar"
  is not a property the bar can have and is not checked.)
* WIDTH  (companion) the widest AUROC interval report.json may carry. The
  interval must contain the measured secret AUROC, so WIDTH is sized from two
  variances: sd_m, the problem-level cluster bootstrap sd (B = 2000, seeded)
  of the canonical probe's AUROC on the secret set, and sd_e, the canonical
  grouped-CV estimate's own standard error, sd(fold_aurocs) / sqrt(folds per
  partition). WIDTH = ceil(4 * sqrt(sd_m^2 + sd_e^2) / 0.01) * 0.01 clamped to
  [0.06, 0.12]. Rows of one problem share X_prompt and therefore a probe's
  score, so the effective sample is the problems, not the rows; resampling
  rows would understate sd_m by about 30 percent on this data. Repeating the
  CV over shuffle seeds stabilises sd(fold_aurocs) but reuses the same
  training problems, so the repeat count does not enter sd_e.
* TOL  the retired point-estimate tolerance, kept for the record. It was the
  larger of 0.05 and twice the seed-to-seed spread of the grouped CV estimate,
  which is split variance on fixed data rather than the measurement's
  variance; the interval gate replaced it.

It writes ``tools/output/calibration.json`` with every intermediate number, so
the controller can see why the constants are what they are.

Two feature leakages are measured on purpose. The four rows of one problem share
the exact same ``X_prompt``, so a random row-wise split leaks a problem's labels
across the train/test boundary and inflates the estimate; a split that keeps a
problem's rows together (grouped on ``problem_id``) does not. The gap between the
two is what the companion's estimate gate is meant to catch: an interval of
width WIDTH centred on the leaky estimate must miss the secret AUROC.

The CV AUROC for a seed is the single AUROC over the pooled out-of-fold scores.
The grouped splits are seeded (``GroupKFold`` with ``shuffle=True``), so each
seed gives a slightly different estimate.
"""
import argparse
import json
import math
import sys
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = (0, 1, 2, 3, 4)
C = 0.3
MAX_ITER = 3000
N_SPLITS = 5
BAR_CLAMP = (0.55, 0.70)
BAR_MARGIN_CONCERN = 0.02  # a side of the flagship bar closer than this is reported
WEAK_PROBE_KEY = "t11_weak_probe"  # pool_report.json entry written by the flagship build
ROWS_PER_PROBLEM = 4
TOL_FLOOR = 0.05

# Companion interval gate.
B_BOOT = 2000
BOOT_SEED = 20261108
WIDTH_SD_MULT = 4.0
WIDTH_CLAMP = (0.06, 0.12)
BAR_COMPANION = 0.55
BAR_SD_MARGIN = 3.0
RANDOM_SCORES_SEED = 20261107  # the random_scores QC variant's seed, so the check is about that row
# The canonical solution's validation scheme, replicated here so sd_e and the
# canonical interval are the ones its report.json carries (same seeds, same
# fold count, same pipeline).
CANONICAL_N_SPLITS = 10
CANONICAL_SEEDS = (0, 1, 2, 3, 4)
INTERVAL_SD_MULT = 2.0

# Paths relative to the repo root.
FLAGSHIP_TRAIN = "grader-hardening/environment/app/env/public/data/train.npz"
POOL_REPORT = "tools/output/pool_report.json"
COMPANION_TRAIN = "honest-probe/environment/app/data/train.npz"
COMPANION_SECRET = "honest-probe/tests/private/test.npz"
OUTPUT = "tools/output/calibration.json"


# ----------------------------------------------------------- rounding helpers

def floor_to_01(x):
    """Floor ``x`` down to a multiple of 0.01 (exact, via Decimal)."""
    return float(Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_FLOOR))


def ceil_to_01(x):
    """Round ``x`` up to a multiple of 0.01 (exact, via Decimal)."""
    return float(Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def round_to_01(x):
    """Round ``x`` half up to a multiple of 0.01 (exact, via Decimal)."""
    return float(Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def bar_from_midpoint(holdout_per_legit, weak_holdout):
    """BAR = round((min honest holdout AUROC + weak-probe holdout AUROC) / 2, 2).

    Rounded half up to 0.01 and clamped to [0.55, 0.70]. Returns (bar, inputs)
    where inputs records the weakest honest item, the weak probe's AUROC, the
    unrounded midpoint and the margin on each side of the clamped bar.
    """
    if not holdout_per_legit:
        raise ValueError("no honest pool items in the pool report")
    weak_holdout = float(weak_holdout)
    if not math.isfinite(weak_holdout):
        raise ValueError(f"weak-probe holdout AUROC must be finite, got {weak_holdout!r}")
    worst_id = min(holdout_per_legit, key=holdout_per_legit.get)
    worst = float(holdout_per_legit[worst_id])
    midpoint = (worst + weak_holdout) / 2.0
    bar = float(min(max(round_to_01(midpoint), BAR_CLAMP[0]), BAR_CLAMP[1]))
    inputs = {
        "min_honest_id": worst_id,
        "min_honest_holdout_auroc": worst,
        "weak_probe_holdout_auroc": weak_holdout,
        "midpoint": midpoint,
        "margin_honest": worst - bar,
        "margin_weak": bar - weak_holdout,
    }
    return bar, inputs


def tol_from_gap_sd(gap_sd):
    """TOL = max(0.05, ceil_to_0.01(2 * gap_sd)). Retired; kept for the record."""
    return float(max(TOL_FLOOR, ceil_to_01(2.0 * gap_sd)))


def width_from_sds(sd_m, sd_e):
    """WIDTH = ceil(4 * sqrt(sd_m^2 + sd_e^2) / 0.01) * 0.01, clamped to [0.06, 0.12].

    sd_m is the problem-level bootstrap sd of the secret-set AUROC (the
    measurement), sd_e the grouped-CV estimate's own standard error. Four
    combined sd is two on each side of the estimate.
    """
    sd_m, sd_e = float(sd_m), float(sd_e)
    for name, v in (("sd_m", sd_m), ("sd_e", sd_e)):
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"{name} must be a finite non-negative number, got {v!r}")
    raw = ceil_to_01(WIDTH_SD_MULT * math.sqrt(sd_m ** 2 + sd_e ** 2))
    return float(min(max(raw, WIDTH_CLAMP[0]), WIDTH_CLAMP[1]))


def interval_half_width(sd_folds, problems_per_fold, folds_per_partition, secret_problems,
                        multiplier=INTERVAL_SD_MULT):
    """Half-width of the solutions' interval (controller ruling W1).

    multiplier * sqrt(sd_folds^2 * problems_per_fold / secret_problems
                      + sd_folds^2 / folds_per_partition):
    the measurement's variance on the secret set plus the CV mean's own
    variance. folds_per_partition is the fold count of one partition of the
    training problems (problems_per_fold / folds_per_partition = 1 / train
    problems); repeats over seeds do not enter.
    """
    sd_folds = float(sd_folds)
    if not math.isfinite(sd_folds) or sd_folds < 0.0:
        raise ValueError(f"sd_folds must be a finite non-negative number, got {sd_folds!r}")
    if problems_per_fold <= 0 or folds_per_partition <= 0 or secret_problems <= 0:
        raise ValueError("problems_per_fold, folds_per_partition and secret_problems must be positive")
    var_measure = sd_folds ** 2 * problems_per_fold / secret_problems
    var_estimate = sd_folds ** 2 / folds_per_partition
    return float(multiplier * math.sqrt(var_measure + var_estimate))


def below_bar_by_sd(bar, point, sd, k=BAR_SD_MARGIN):
    """True when a measurement sits at least k of its own sd below the bar."""
    return bool(bar - float(point) >= k * float(sd))


def above_bar_by_sd(bar, point, sd, k=BAR_SD_MARGIN):
    """True when a measurement sits at least k of its own sd above the bar."""
    return bool(float(point) - bar >= k * float(sd))


# ------------------------------------------------------------ cluster bootstrap

def cluster_bootstrap_auroc(y, scores, groups, auroc, n_boot=B_BOOT, seed=BOOT_SEED):
    """Problem-level (cluster) bootstrap of the AUROC of fixed scores.

    Clusters (problems) are resampled with replacement, every row of a drawn
    cluster comes along, and the AUROC is recomputed n_boot times with a seeded
    generator. Rows of one problem share X_prompt and so a probe's score, so
    they are not independent; resampling clusters respects that. A resample
    whose labels hold a single class has no AUROC; it is skipped and counted.

    Returns a dict: point (AUROC on the full data), sd (ddof=1 over the valid
    resamples), p2_5, p97_5, n_boot, n_valid, n_degenerate, n_clusters,
    n_rows, seed.
    """
    y = np.asarray(y).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if not (y.shape == scores.shape == groups.shape):
        raise ValueError("y, scores and groups must have the same length")
    if y.size == 0:
        raise ValueError("empty input")
    if int(n_boot) < 2:
        raise ValueError("n_boot must be at least 2")
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]
    bounds = np.searchsorted(sorted_inv, np.arange(uniq.size + 1))
    members = [order[bounds[i]:bounds[i + 1]] for i in range(uniq.size)]

    rng = np.random.default_rng(seed)
    vals = []
    degenerate = 0
    for _ in range(int(n_boot)):
        draw = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([members[c] for c in draw])
        yy = y[idx]
        if yy.min() == yy.max():
            degenerate += 1
            continue
        vals.append(float(auroc(yy, scores[idx])))
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size < 2:
        raise ValueError(f"only {vals.size} valid resamples out of {n_boot}; "
                         "the data has too few clusters with both classes")
    return {
        "point": float(auroc(y, scores)),
        "sd": float(vals.std(ddof=1)),
        "p2_5": float(np.percentile(vals, 2.5)),
        "p97_5": float(np.percentile(vals, 97.5)),
        "n_boot": int(n_boot),
        "n_valid": int(vals.size),
        "n_degenerate": int(degenerate),
        "n_clusters": int(uniq.size),
        "n_rows": int(y.size),
        "seed": int(seed),
    }


# --------------------------------------------------------------- model and cv

def _model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=MAX_ITER),
    )


def _oof_auroc(X, y, splits, auroc):
    """Pooled out-of-fold AUROC: one score per row, then a single AUROC."""
    oof = np.full(y.shape[0], np.nan, dtype=np.float64)
    for train_idx, test_idx in splits:
        model = _model()
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.decision_function(X[test_idx])
    if not np.isfinite(oof).all():
        raise ValueError("some rows received no out-of-fold score")
    return float(auroc(y, oof))


def grouped_cv_auroc(X, y, groups, auroc, seeds=SEEDS):
    """Mean over seeds of the grouped (problem-disjoint) pooled OOF AUROC."""
    per_seed = []
    for seed in seeds:
        splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        per_seed.append(_oof_auroc(X, y, splitter.split(X, y, groups), auroc))
    return float(np.mean(per_seed)), per_seed


def row_cv_auroc(X, y, auroc, seeds=SEEDS):
    """Mean over seeds of the random row-wise (leaky) pooled OOF AUROC."""
    per_seed = []
    for seed in seeds:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        per_seed.append(_oof_auroc(X, y, splitter.split(X, y), auroc))
    return float(np.mean(per_seed)), per_seed


def canonical_fold_aurocs(X, y, groups, auroc, n_splits=CANONICAL_N_SPLITS, seeds=CANONICAL_SEEDS):
    """One AUROC per GroupKFold validation fold over every seed (the canonical scheme).

    Returns (fold_aurocs, problems_per_fold, n_skipped). Not pooled: the spread
    of these values is what sizes the solutions' interval. A validation fold
    holding a single class has no AUROC and is skipped and counted; that
    cannot happen with 70 problems per fold, only on tiny synthetic data.
    """
    folds, sizes, skipped = [], [], 0
    for seed in seeds:
        splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in splitter.split(X, y, groups):
            if np.unique(y[test_idx]).size != 2:
                skipped += 1
                continue
            model = _model()
            model.fit(X[train_idx], y[train_idx])
            folds.append(float(auroc(y[test_idx], model.decision_function(X[test_idx]))))
            sizes.append(np.unique(groups[test_idx]).size)
    if len(folds) < 2:
        raise ValueError(f"only {len(folds)} usable validation folds; need at least 2")
    return folds, float(np.mean(sizes)), skipped


# --------------------------------------------------------------- data loading

def _load(path):
    with np.load(path, allow_pickle=False) as d:
        return (
            d["X_prompt"].astype(np.float64),
            d["X_reason"].astype(np.float64),
            d["y"].astype(np.int64),
            d["problem_id"].astype(np.int64),
        )


# ----------------------------------------------------------------- the metrics

def _boot_summary(boot, keys=("sd", "p2_5", "p97_5")):
    return {k: boot[k] for k in keys}


def compute_calibration(repo):
    """Compute every calibration quantity. No checks, no file writes."""
    repo = Path(repo)
    sys.path.insert(0, str((repo / "tools" / "common").resolve()))
    from auroc import auroc  # vendored, numpy-only rank AUROC

    # Flagship public train: honest grouped CV vs leaky row CV.
    Xp, Xr, y, pid = _load(repo / FLAGSHIP_TRAIN)
    honest_cv_prompt, _ = grouped_cv_auroc(Xp, y, pid, auroc)
    honest_cv_reason, _ = grouped_cv_auroc(Xr, y, pid, auroc)
    honest_cv_concat, _ = grouped_cv_auroc(
        np.concatenate([Xp, Xr], axis=1), y, pid, auroc)
    leaky_row_cv_prompt, _ = row_cv_auroc(Xp, y, auroc)

    # Legit pool holdout AUROCs and the weak probe's: copied from the measured
    # pool report, which the flagship build writes.
    pool = json.loads((repo / POOL_REPORT).read_text(encoding="utf-8"))
    holdout_auroc_per_legit = {
        k: float(v["holdout_auroc"]) for k, v in pool.items()
        if k.startswith("L") and isinstance(v, dict)
    }
    if WEAK_PROBE_KEY not in pool or "holdout_auroc" not in pool[WEAK_PROBE_KEY]:
        raise ValueError(f"{POOL_REPORT} has no {WEAK_PROBE_KEY}.holdout_auroc entry; "
                         "rebuild the pools first")
    weak_holdout = float(pool[WEAK_PROBE_KEY]["holdout_auroc"])
    holdout_rows = int(pool["holdout_rows"]) if "holdout_rows" in pool else None

    # Companion: honest grouped estimate and leaky row estimate vs the secret
    # AUROC a probe trained on all the companion train data reaches.
    cXp, _cXr, cy, cpid = _load(repo / COMPANION_TRAIN)
    sXp, _sXr, sy, spid = _load(repo / COMPANION_SECRET)
    secret_model = _model()
    secret_model.fit(cXp, cy)
    secret_scores = secret_model.decision_function(sXp)
    companion_secret_auroc = float(auroc(sy, secret_scores))

    companion_grouped_estimate, grouped_per_seed = grouped_cv_auroc(cXp, cy, cpid, auroc)
    companion_leaky_estimate, _ = row_cv_auroc(cXp, cy, auroc)
    gaps = [est - companion_secret_auroc for est in grouped_per_seed]
    companion_estimate_gap_sd = float(np.std(gaps))  # population sd over seeds

    # Problem-level cluster bootstrap of the secret measurement (sd_m), and of
    # the no-signal scores the companion bar must reject (the random_scores QC
    # variant's exact scores).
    boot_secret = cluster_bootstrap_auroc(sy, secret_scores, spid, auroc)
    random_scores = np.random.default_rng(RANDOM_SCORES_SEED).standard_normal(sy.size)
    boot_random = cluster_bootstrap_auroc(sy, random_scores, spid, auroc)

    # The canonical solution's own validation, replicated: per-fold AUROCs
    # under its scheme give sd(fold_aurocs), the estimate's standard error
    # sd_e, and the interval its report.json carries.
    canon_folds, canon_ppf, canon_skipped = canonical_fold_aurocs(cXp, cy, cpid, auroc)
    canon_folds_arr = np.asarray(canon_folds, dtype=np.float64)
    canon_mean = float(canon_folds_arr.mean())
    canon_sd = float(canon_folds_arr.std(ddof=1))
    sd_e = canon_sd / math.sqrt(CANONICAL_N_SPLITS)
    secret_problems = int(np.unique(spid).size)
    canon_half = interval_half_width(canon_sd, canon_ppf, CANONICAL_N_SPLITS, secret_problems)
    canon_lo, canon_hi = canon_mean - canon_half, canon_mean + canon_half

    bar, bar_inputs = bar_from_midpoint(holdout_auroc_per_legit, weak_holdout)
    bar_inputs["holdout_rows"] = holdout_rows
    bar_inputs["holdout_problems"] = None if holdout_rows is None else holdout_rows // ROWS_PER_PROBLEM
    tol = tol_from_gap_sd(companion_estimate_gap_sd)
    width = width_from_sds(boot_secret["sd"], sd_e)

    return {
        "honest_cv_prompt": honest_cv_prompt,
        "honest_cv_reason": honest_cv_reason,
        "honest_cv_concat": honest_cv_concat,
        "leaky_row_cv_prompt": leaky_row_cv_prompt,
        "holdout_auroc_per_legit": holdout_auroc_per_legit,
        "flagship_bar_inputs": bar_inputs,
        "companion_secret_auroc": companion_secret_auroc,
        "companion_grouped_estimate": companion_grouped_estimate,
        "companion_estimate_gap_sd": companion_estimate_gap_sd,
        "companion_leaky_estimate": companion_leaky_estimate,
        "companion_leaky_gap": companion_leaky_estimate - companion_secret_auroc,
        "companion_secret_bootstrap": {
            "n_boot": boot_secret["n_boot"], "seed": boot_secret["seed"],
            "n_problems": boot_secret["n_clusters"], "n_rows": boot_secret["n_rows"],
            "n_degenerate": boot_secret["n_degenerate"],
            **_boot_summary(boot_secret),
            "clears_bar_by_3sd": above_bar_by_sd(BAR_COMPANION, boot_secret["point"],
                                                 boot_secret["sd"]),
        },
        "companion_random_scores": {
            "seed": RANDOM_SCORES_SEED, "auroc": boot_random["point"],
            **_boot_summary(boot_random),
            "below_bar_by_3sd": below_bar_by_sd(BAR_COMPANION, boot_random["point"],
                                                boot_random["sd"]),
        },
        "companion_canonical_folds": {
            "n_splits": CANONICAL_N_SPLITS, "seeds": list(CANONICAL_SEEDS),
            "n_folds": int(canon_folds_arr.size), "n_skipped": int(canon_skipped),
            "problems_per_fold": canon_ppf,
            "train_problems": int(np.unique(cpid).size),
            "mean": canon_mean, "sd": canon_sd, "sd_e": sd_e,
        },
        "companion_canonical_interval": {
            "secret_problems": secret_problems, "half_width": canon_half,
            "lo": canon_lo, "hi": canon_hi, "width": 2.0 * canon_half,
            "contains": bool(canon_lo <= companion_secret_auroc <= canon_hi),
            "margin_lo": companion_secret_auroc - canon_lo,
            "margin_hi": canon_hi - companion_secret_auroc,
        },
        "sd_m": boot_secret["sd"],
        "sd_e": sd_e,
        "BAR": round(bar, 2),
        "BAR_companion": BAR_COMPANION,
        "WIDTH": round(width, 2),
        "TOL": round(tol, 2),
    }


# ------------------------------------------------------------------- the gates

def check_loud_failures(result):
    """Return a list of loud-failure messages; empty means all gates hold.

    A non-empty list means calibration.json must NOT be written.
    """
    failures = []
    bar = result["BAR"]
    leak = result["leaky_row_cv_prompt"] - result["honest_cv_prompt"]
    if leak < 0.10:
        failures.append(
            f"leakage too small: leaky_row_cv_prompt - honest_cv_prompt = {leak:.4f} < 0.10")
    for name, val in result["holdout_auroc_per_legit"].items():
        if not val > bar:
            failures.append(f"legit {name} holdout AUROC {val:.4f} does not clear BAR {bar:.2f}")
    inputs = result.get("flagship_bar_inputs")
    if inputs is not None and not inputs["weak_probe_holdout_auroc"] < bar:
        failures.append(
            f"weak probe holdout AUROC {inputs['weak_probe_holdout_auroc']:.4f} is not below "
            f"BAR {bar:.2f}")
    boot = result.get("companion_secret_bootstrap")
    if boot is not None:
        secret, sd = result["companion_secret_auroc"], boot["sd"]
        if not boot.get("clears_bar_by_3sd", above_bar_by_sd(BAR_COMPANION, secret, sd)):
            failures.append(
                f"companion_secret_auroc {secret:.4f} does not clear BAR_companion "
                f"{BAR_COMPANION} by 3 sd ({3 * sd:.4f})")
    return failures


def check_trap_robustness(result):
    """Return a list of trap-robustness concern messages; empty means it holds.

    These do NOT gate writing calibration.json; a non-empty list is reported as
    DONE_WITH_CONCERNS so the controller can rule on the constants. The checks
    are the interval gate's: no legal interval (width at most WIDTH) around
    the leaky estimate can contain the secret AUROC; the canonical's own
    interval contains it and is no wider than WIDTH; and random scores sit at
    least 3 sd below BAR_companion.
    """
    concerns = []
    inputs = result.get("flagship_bar_inputs")
    if inputs is not None:
        for side in ("margin_honest", "margin_weak"):
            if inputs[side] < BAR_MARGIN_CONCERN:
                concerns.append(
                    f"flagship bar {side} = {inputs[side]:.4f} is below {BAR_MARGIN_CONCERN} "
                    f"(about one sd of a 600-problem holdout measurement)")
    width = result["WIDTH"]
    secret = result["companion_secret_auroc"]
    naive_gap = abs(result["companion_leaky_estimate"] - secret)
    honest_gap = abs(result["companion_grouped_estimate"] - secret)
    if not naive_gap > width:
        concerns.append(
            f"naive estimate not caught: |leaky - secret| = {naive_gap:.4f} is not > "
            f"WIDTH = {width:.4f}, so an interval of legal width around it could contain "
            f"the secret AUROC")
    if not honest_gap < width / 2.0:
        concerns.append(
            f"honest estimate not reliably covered: |grouped - secret| = {honest_gap:.4f} "
            f"is not < WIDTH / 2 = {width / 2.0:.4f}")
    canon = result.get("companion_canonical_interval")
    if canon is not None:
        if not canon["contains"]:
            concerns.append(
                f"the canonical interval [{canon['lo']:.4f}, {canon['hi']:.4f}] does not "
                f"contain the secret AUROC {secret:.4f}")
        if canon["width"] > width + 1e-9:
            concerns.append(
                f"the canonical interval is {canon['width']:.4f} wide, more than WIDTH {width:.4f}")
    entry = result.get("companion_random_scores")
    if entry is not None and not entry["below_bar_by_3sd"]:
        concerns.append(
            f"random scores measure {entry['auroc']:.4f} with sd {entry['sd']:.4f}: not at "
            f"least 3 sd below BAR_companion {BAR_COMPANION}")
    return concerns


# ------------------------------------------------------------------------ main

def _print_result(result):
    print("Calibration numbers:")
    for key, val in result.items():
        if isinstance(val, dict):
            parts = []
            for k, v in val.items():
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
            print(f"  {key}: " + ", ".join(parts))
        else:
            print(f"  {key}: {val}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Calibrate the flagship BAR, BAR_companion and WIDTH from the data.")
    ap.add_argument("--repo", default=".", help="repo root (default: .)")
    args = ap.parse_args(argv)
    repo = Path(args.repo)

    result = compute_calibration(repo)
    _print_result(result)

    failures = check_loud_failures(result)
    if failures:
        print("\nLOUD FAILURE (calibration.json NOT written):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        raise SystemExit(1)

    out_path = repo / OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWrote {out_path}")

    concerns = check_trap_robustness(result)
    boot = result["companion_secret_bootstrap"]
    inputs = result["flagship_bar_inputs"]
    print(f"\nBAR = {result['BAR']} (midpoint of {inputs['min_honest_id']} "
          f"{inputs['min_honest_holdout_auroc']:.4f} and weak probe "
          f"{inputs['weak_probe_holdout_auroc']:.4f}; margins {inputs['margin_honest']:+.4f} / "
          f"{inputs['margin_weak']:+.4f})    BAR_companion = {result['BAR_companion']}    "
          f"WIDTH = {result['WIDTH']}    (TOL = {result['TOL']}, retired)")
    print(f"Secret measurement: {result['companion_secret_auroc']:.4f}, cluster-bootstrap sd_m "
          f"{boot['sd']:.4f} over {boot['n_problems']} problems, 95 percent "
          f"[{boot['p2_5']:.4f}, {boot['p97_5']:.4f}]; estimate sd_e {result['sd_e']:.4f}")
    canon = result["companion_canonical_interval"]
    print(f"Canonical: grouped fold mean {result['companion_canonical_folds']['mean']:.4f}, "
          f"interval [{canon['lo']:.4f}, {canon['hi']:.4f}] (width {canon['width']:.4f}), "
          f"contains={canon['contains']} margins lo {canon['margin_lo']:+.4f} hi {canon['margin_hi']:+.4f}")
    print(f"Trap robustness: naive |leaky - secret| = "
          f"{abs(result['companion_leaky_estimate'] - result['companion_secret_auroc']):.4f} "
          f"(need > WIDTH {result['WIDTH']:.4f}); honest |grouped - secret| = "
          f"{abs(result['companion_grouped_estimate'] - result['companion_secret_auroc']):.4f} "
          f"(need < WIDTH / 2 = {result['WIDTH'] / 2.0:.4f})")
    if concerns:
        print("\nDONE_WITH_CONCERNS: the companion gates do not cleanly separate on this data.")
        for c in concerns:
            print(f"  - {c}")
    else:
        print("\nDONE: all loud-failure and trap-robustness checks pass.")


if __name__ == "__main__":
    main()
