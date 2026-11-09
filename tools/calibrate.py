"""Calibrate the verifier constants from the data.

Run as::

    python tools/calibrate.py --repo .

The script measures, on the committed data, the quantities that set the
constants used by the Harbor tasks:

* BAR  (flagship) the AUROC a probe must clear in grader-hardening. Derived
  from the weakest legit pool solution's holdout AUROC, with a 0.04 margin
  below it, floored to 0.01 and clamped to [0.55, 0.70].
* BAR_companion  fixed at 0.55 for honest-probe: a floor well above chance
  that is decoupled from where the reference probe lands. Recorded with it is
  the check that random scores and a deliberately weak probe (the canonical
  pipeline fit on 10 problems, 40 rows) stay below it by at least 3 standard
  deviations of their own measurement, and the loud check that the canonical
  probe clears it by at least 3 standard deviations.
* WIDTH  (companion) the widest AUROC interval report.json may carry. The
  interval must contain the measured secret AUROC, so WIDTH is sized from the
  measurement's own uncertainty: a problem-level cluster bootstrap (B = 2000,
  seeded) of the canonical probe's AUROC on the secret set gives sd, and
  WIDTH = ceil(4 * sd / 0.01) * 0.01 clamped to [0.06, 0.12]. Rows of one
  problem share X_prompt and therefore a probe's score, so the effective
  sample is the problems, not the rows; resampling rows would understate sd
  by about 30 percent on this data.
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
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
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
MARGIN = 0.04
BAR_CLAMP = (0.55, 0.70)
TOL_FLOOR = 0.05

# Companion interval gate.
B_BOOT = 2000
BOOT_SEED = 20261108
WIDTH_SD_MULT = 4.0
WIDTH_CLAMP = (0.06, 0.12)
BAR_COMPANION = 0.55
BAR_SD_MARGIN = 3.0
RANDOM_SCORES_SEED = 20261107  # the random_scores QC variant's seed, so the check is about that row
WEAK_PROBE_PROBLEMS = 10  # 40 rows: the deliberately weak probe
WEAK_PROBE_SEED = 0
WEAK_PROBE_CONTEXT_SEEDS = (0, 1, 2, 3, 4)  # recorded for context; the check uses WEAK_PROBE_SEED

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


def bar_from_holdout(holdout_per_legit):
    """BAR = floor_to_0.01(min(holdout) - 0.04), clamped to [0.55, 0.70]."""
    worst = min(holdout_per_legit.values())
    raw = floor_to_01(worst - MARGIN)
    return float(min(max(raw, BAR_CLAMP[0]), BAR_CLAMP[1]))


def tol_from_gap_sd(gap_sd):
    """TOL = max(0.05, ceil_to_0.01(2 * gap_sd)). Retired; kept for the record."""
    return float(max(TOL_FLOOR, ceil_to_01(2.0 * gap_sd)))


def width_from_sd(sd):
    """WIDTH = ceil(4 * sd / 0.01) * 0.01, clamped to [0.06, 0.12].

    sd is the problem-level bootstrap standard deviation of the secret-set
    AUROC. Four sd is two sd on each side of the estimate.
    """
    sd = float(sd)
    if not math.isfinite(sd) or sd < 0.0:
        raise ValueError(f"sd must be a finite non-negative number, got {sd!r}")
    raw = ceil_to_01(WIDTH_SD_MULT * sd)
    return float(min(max(raw, WIDTH_CLAMP[0]), WIDTH_CLAMP[1]))


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


def weak_probe_scores(X_train, y_train, pid_train, X_eval,
                      n_problems=WEAK_PROBE_PROBLEMS, seed=WEAK_PROBE_SEED):
    """Scores of the canonical pipeline fit on a seeded draw of n_problems problems.

    Returns (scores, problems_used, rows_used). The draw is retried with the
    next seed (bounded) when it holds a single class, which cannot be fit.
    """
    problems = np.unique(pid_train)
    if problems.size < n_problems:
        raise ValueError(f"need at least {n_problems} problems, have {problems.size}")
    for attempt in range(20):
        rng = np.random.default_rng(seed + attempt)
        chosen = rng.choice(problems, size=n_problems, replace=False)
        mask = np.isin(pid_train, chosen)
        if np.unique(y_train[mask]).size == 2:
            model = _model().fit(X_train[mask], y_train[mask])
            return model.decision_function(X_eval), sorted(int(p) for p in chosen), int(mask.sum())
    raise ValueError("no draw of weak-probe problems held both classes")


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

    # Legit pool holdout AUROCs: copied from the measured pool report.
    pool = json.loads((repo / POOL_REPORT).read_text(encoding="utf-8"))
    holdout_auroc_per_legit = {
        k: float(v["holdout_auroc"]) for k, v in pool.items() if k.startswith("L")
    }

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

    # Problem-level cluster bootstrap of the secret measurement, and of the two
    # things the companion bar must reject: random scores (the random_scores
    # QC variant's exact scores) and a weak probe fit on 10 problems.
    boot_secret = cluster_bootstrap_auroc(sy, secret_scores, spid, auroc)
    random_scores = np.random.default_rng(RANDOM_SCORES_SEED).standard_normal(sy.size)
    boot_random = cluster_bootstrap_auroc(sy, random_scores, spid, auroc)
    weak_scores, weak_problems, weak_rows = weak_probe_scores(cXp, cy, cpid, sXp)
    boot_weak = cluster_bootstrap_auroc(sy, weak_scores, spid, auroc)
    weak_by_seed = {
        str(seed): float(auroc(sy, weak_probe_scores(cXp, cy, cpid, sXp, seed=seed)[0]))
        for seed in WEAK_PROBE_CONTEXT_SEEDS
    }

    bar = bar_from_holdout(holdout_auroc_per_legit)
    tol = tol_from_gap_sd(companion_estimate_gap_sd)
    width = width_from_sd(boot_secret["sd"])

    return {
        "honest_cv_prompt": honest_cv_prompt,
        "honest_cv_reason": honest_cv_reason,
        "honest_cv_concat": honest_cv_concat,
        "leaky_row_cv_prompt": leaky_row_cv_prompt,
        "holdout_auroc_per_legit": holdout_auroc_per_legit,
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
        "companion_weak_probe": {
            "seed": WEAK_PROBE_SEED, "train_problems": weak_problems, "train_rows": weak_rows,
            "auroc": boot_weak["point"], **_boot_summary(boot_weak),
            "auroc_by_seed": weak_by_seed,
            "below_bar_by_3sd": below_bar_by_sd(BAR_COMPANION, boot_weak["point"],
                                                boot_weak["sd"]),
        },
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
        if val < bar + MARGIN:
            failures.append(
                f"legit {name} holdout AUROC {val:.4f} < BAR + 0.04 = {bar + MARGIN:.4f}")
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
    the leaky estimate can contain the secret AUROC; an honest interval of
    width WIDTH centred on the grouped estimate does contain it; and the two
    no-signal measurements sit at least 3 sd below BAR_companion.
    """
    concerns = []
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
    for key, label in (("companion_random_scores", "random scores"),
                       ("companion_weak_probe", "the weak probe")):
        entry = result.get(key)
        if entry is not None and not entry["below_bar_by_3sd"]:
            concerns.append(
                f"{label} measure {entry['auroc']:.4f} with sd {entry['sd']:.4f}: not at "
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
    ap = argparse.ArgumentParser(description="Calibrate BAR, BAR_companion and WIDTH from the data.")
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
    print(f"\nBAR = {result['BAR']}    BAR_companion = {result['BAR_companion']}    "
          f"WIDTH = {result['WIDTH']}    (TOL = {result['TOL']}, retired)")
    print(f"Secret measurement: {result['companion_secret_auroc']:.4f}, cluster-bootstrap sd "
          f"{boot['sd']:.4f} over {boot['n_problems']} problems, 95 percent "
          f"[{boot['p2_5']:.4f}, {boot['p97_5']:.4f}]")
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
