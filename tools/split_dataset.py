"""Allocate problems of activations_full.npz, plus the extra set generated with offsets, to the two
tasks' data files, and record the split sizes in dataset_stats.json."""
import argparse
import json
from pathlib import Path

import numpy as np

SEED = 7
COMPANION_SEED = 11  # shuffles companion train rows so ungrouped CV splits leak
N_TRAIN = 2100
PATHS = {
    "train": "grader-hardening/environment/app/env/public/data/train.npz",
    "test": "grader-hardening/environment/app/env/public/data/test.npz",
    "holdout": "grader-hardening/environment/app/holdout/holdout.npz",
    "companion_train": "honest-probe/environment/app/data/train.npz",
    "secret": "honest-probe/tests/private/test.npz",
}
# problem_id intervals [lo, hi); the original run covers 0 to 999, the extra run 1000 to 2199
RANGES = {
    "public": [(0, 700)],
    "holdout": [(700, 850), (1000, 1450)],
    "secret": [(850, 1000), (1450, 2200)],
}
STATS_SECTIONS = ("original", "extra", "combined", "splits")
MAX_RUNS = 4  # problem_id_runs is reported only for splits with at most this many intervals


def _take(full: dict, mask_or_idx) -> dict:
    return {k: np.asarray(v)[mask_or_idx] for k, v in full.items()}


def _in_ranges(pid: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(len(pid), dtype=bool)
    for lo, hi in ranges:
        mask |= (pid >= lo) & (pid < hi)
    return mask


def combine(full: dict, extra: dict | None) -> dict:
    """Concatenate the original and extra arrays; their problem_id and row_id sets must be disjoint."""
    full = {k: np.asarray(v) for k, v in full.items()}
    if extra is None:
        return full
    extra = {k: np.asarray(v) for k, v in extra.items()}
    assert set(full) == set(extra), (sorted(full), sorted(extra))
    for key in ("problem_id", "row_id"):
        assert not set(full[key].tolist()) & set(extra[key].tolist()), f"{key} overlap between the runs"
    return {k: np.concatenate([full[k], extra[k]]) for k in full}


def allocate(full: dict, extra: dict | None = None) -> dict[str, dict]:
    full = combine(full, extra)
    pid = full["problem_id"]
    covered = _in_ranges(pid, RANGES["public"] + RANGES["holdout"] + RANGES["secret"])
    assert covered.all(), f"problem_id outside every range: {sorted(set(pid[~covered].tolist()))[:5]}"
    by_row = lambda m: np.flatnonzero(m)[np.argsort(full["row_id"][m], kind="stable")]
    public = by_row(_in_ranges(pid, RANGES["public"]))
    perm = np.random.default_rng(SEED).permutation(len(public))
    shuffled = public[perm]
    companion = public[np.random.default_rng(COMPANION_SEED).permutation(len(public))]
    out = {
        PATHS["train"]: _take(full, shuffled[:N_TRAIN]),
        PATHS["test"]: _take(full, shuffled[N_TRAIN:]),
        PATHS["holdout"]: _take(full, by_row(_in_ranges(pid, RANGES["holdout"]))),
        PATHS["companion_train"]: _take(full, companion),
        PATHS["secret"]: _take(full, by_row(_in_ranges(pid, RANGES["secret"]))),
    }
    tr, te = out[PATHS["train"]], out[PATHS["test"]]
    assert not set(tr["row_id"].tolist()) & set(te["row_id"].tolist())
    ho, se = out[PATHS["holdout"]], out[PATHS["secret"]]
    assert not set(ho["problem_id"].tolist()) & set(se["problem_id"].tolist())
    return out


def _runs(values) -> list[list[int]]:
    """Maximal runs of consecutive integers in a sorted unique array, as [first, last] pairs."""
    values = np.asarray(values)
    if len(values) == 0:
        return []
    breaks = np.flatnonzero(np.diff(values) != 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(values) - 1]])
    return [[int(values[s]), int(values[e])] for s, e in zip(starts, ends)]


def _summary(arrays: dict) -> dict:
    """Row and problem counts, problem_id range and class balance of one array set. problem_id_runs
    (maximal consecutive intervals) is listed only when there are at most MAX_RUNS of them; the
    row-split public files fragment into hundreds."""
    pid = np.unique(arrays["problem_id"])
    runs = _runs(pid)
    summary = {
        "rows": int(len(arrays["y"])),
        "problems": int(len(pid)),
        "problem_id_range": [int(pid.min()), int(pid.max())],
    }
    if len(runs) <= MAX_RUNS:
        summary["problem_id_runs"] = runs
    summary["positives"] = int(arrays["y"].sum())
    summary["positive_rate"] = round(float(arrays["y"].mean()), 4)
    return summary


def split_stats(full: dict, extra: dict | None, out: dict[str, dict]) -> dict:
    """The "combined" and "splits" sections of dataset_stats.json."""
    combined = combine(full, extra)
    return {
        "combined": _summary(combined),
        "splits": {name: {"path": PATHS[name], **_summary(out[PATHS[name]])} for name in PATHS},
    }


def update_stats(stats_path: Path, extra_stats_path: Path | None, sections: dict) -> dict:
    """Rewrite dataset_stats.json as {original, extra, combined, splits}. A file in the generator's
    flat format (one run's stats at top level) is moved under "original"."""
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    original = stats.get("original") or {k: v for k, v in stats.items() if k not in STATS_SECTIONS}
    new = {"original": original}
    extra = json.loads(extra_stats_path.read_text(encoding="utf-8")) if extra_stats_path else stats.get("extra")
    if extra is not None:
        new["extra"] = extra
    new.update(sections)
    stats_path.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    return new


def _load(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="activations_full.npz of the original run")
    ap.add_argument("--extra", default=None, help="activations .npz of the run generated with id offsets")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--stats", type=Path, default=None, help="dataset_stats.json to rewrite with the split sizes")
    ap.add_argument("--extra-stats", type=Path, default=None,
                    help="the extra run's stats json, copied into the \"extra\" section of --stats")
    a = ap.parse_args()
    full = _load(a.src)
    extra = _load(a.extra) if a.extra else None
    out = allocate(full, extra)
    for rel, arrays in out.items():
        dst = Path(a.repo) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **arrays)
        print(f"{rel}: {len(arrays['y'])} rows, {len(np.unique(arrays['problem_id']))} problems")
    if a.stats:
        update_stats(a.stats, a.extra_stats, split_stats(full, extra, out))
        print(f"{a.stats}: updated")


if __name__ == "__main__":
    main()
