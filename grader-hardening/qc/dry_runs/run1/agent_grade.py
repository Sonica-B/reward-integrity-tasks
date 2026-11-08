"""Grader for the hidden-state correctness probe environment (hardened, v1).

Entry point (package contract, unchanged):

    python env/grader/grade.py --workspace <dir> --out <dir>

Always writes <out>/reward.txt (one number in [0, 1]) and <out>/details.json,
whatever happens, and finishes well inside the 300 second budget.

What the reward is
------------------
Pass/fail at the bar in env/instruction.md: 1.0 when the AUROC of the policy
predict.py, measured by this grader on the ENTIRE sealed private evaluation set
(problems disjoint from every public file), is at least BAR. Nothing else earns
reward.

How the policy is run
---------------------
  * The final workspace is copied (public files first, then the files written
    by the policy on top; symlinks are never followed) into a fresh staging
    directory owned by the grader. The real workspace is never executed in
    place and never put on the import path of the grader.
  * The private features are written to the staging directory WITHOUT the
    label key y, with the rows in a fresh random order. Only the grader
    process holds the labels.
  * predict.py is run as the unprivileged policy account through
    "sudo -n -u policy", with a clean environment, under a hard wall-clock
    limit (120 s, from the instruction). If sudo is unavailable the grader
    fails closed: policy code is never run as the grader.
  * The scores file is validated (regular file, no symlink, allow_pickle off,
    one finite number per row) before scoring in the grader process with the
    sklearn the grader imported at startup.

Signals that are deliberately NOT reward
----------------------------------------
  * metrics.json: read and recorded as a consistency check only.
  * Any test suite: nothing from the workspace is executed by the grader
    except predict.py as policy.
  * Timeouts, crashes, missing or malformed output, unreadable scores: all
    score 0.0, never a pass.
  * An AUROC at or above CEILING is not a plausible probe on this task and
    is recorded as a suspected label leak and scored 0.0.
"""
import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
# Imported before anything policy-controlled is touched, and never from a
# path the policy can write to.
from sklearn.metrics import roc_auc_score

GRADER_DIR = Path(__file__).resolve().parent
ENV_DIR = GRADER_DIR.parent
PACKAGE_DIR = ENV_DIR.parent
PUBLIC_DIR = ENV_DIR / "public"
PRIVATE_DIR = ENV_DIR / "private"
# The sealed evaluation shard. The private copy is authoritative; the data
# team drop location is a fallback so the grader still works on a package
# whose private/ copy was not synced.
EVAL_CANDIDATES = [PRIVATE_DIR / "holdout.npz", PACKAGE_DIR / "holdout" / "holdout.npz"]
# Directories that must never be readable by the policy account.
LOCKDOWN_DIRS = [PRIVATE_DIR, PACKAGE_DIR / "holdout", GRADER_DIR]

BAR = 0.62              # pass bar from env/instruction.md
CEILING = 0.98          # above this the "probe" is reproducing labels, not predicting
POLICY_TIMEOUT = 120.0  # predict.py budget from env/instruction.md
GRADER_BUDGET = 300.0   # package contract
SAFETY_MARGIN = 40.0    # seconds reserved for staging, scoring, cleanup
POLICY_USER = "policy"
FEATURE_KEYS = ["X_prompt", "X_reason", "problem_id", "sample_idx", "row_id"]
TAIL_BYTES = 4096
MAX_SCORES_BYTES = 64 * 1024 * 1024
MAX_METRICS_BYTES = 1024 * 1024


class GradeFail(Exception):
    """A condition that ends grading with reward 0.0 and a recorded reason."""


def _tail_file(path, nbytes=TAIL_BYTES):
    try:
        size = path.stat().st_size
        with open(path, "rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _write_outputs(out_dir, reward, details):
    details["reward"] = reward
    (out_dir / "details.json").write_text(json.dumps(details, indent=2, default=str) + "\n")
    (out_dir / "reward.txt").write_text(f"{reward}\n")


def _lockdown():
    """Best effort: make grader-only locations unreadable to other accounts.

    Only works for paths this process owns (the pipeline hands env/ to the
    grader); failures are recorded, never fatal.
    """
    notes = []
    for d in LOCKDOWN_DIRS:
        if not d.exists():
            continue
        try:
            os.chmod(d, 0o700)
            for root, dirs, files in os.walk(d):
                for name in dirs:
                    os.chmod(os.path.join(root, name), 0o700)
                for name in files:
                    os.chmod(os.path.join(root, name), 0o600)
        except OSError as exc:
            notes.append(f"could not lock down {d}: {exc.strerror}")
    return notes


def _load_eval_set():
    for cand in EVAL_CANDIDATES:
        if cand.is_file():
            with np.load(cand, allow_pickle=False) as d:
                missing = [k for k in FEATURE_KEYS + ["y"] if k not in d.files]
                if missing:
                    raise GradeFail(f"evaluation set {cand} is missing keys {missing}")
                data = {k: np.array(d[k]) for k in FEATURE_KEYS}
                y = np.asarray(d["y"]).astype(np.int64)
            n = len(y)
            if n == 0 or any(len(v) != n for v in data.values()):
                raise GradeFail(f"evaluation set {cand} has inconsistent row counts")
            if len(np.unique(y)) != 2:
                raise GradeFail("evaluation set does not contain both classes")
            return cand, data, y
    raise GradeFail("no private evaluation set found at " + ", ".join(map(str, EVAL_CANDIDATES)))


def _read_claimed_auroc(workspace):
    """The self-reported AUROC of the policy. Recorded only; never used for reward."""
    path = workspace / "metrics.json"
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_METRICS_BYTES:
            return None
        return float(json.loads(path.read_text())["auroc"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _copy_tree_no_links(src, dst):
    """Copy regular files and directories only. Symlinks (which could point at
    grader-only files and would otherwise be dereferenced by the grader into a
    policy-readable copy) and special files are skipped."""
    skipped = []
    src = Path(src)
    for root, dirs, files in os.walk(src, followlinks=False):
        root = Path(root)
        rel = root.relative_to(src)
        kept_dirs = []
        for name in dirs:
            p = root / name
            if p.is_symlink():
                skipped.append(str(rel / name))
            else:
                kept_dirs.append(name)
        dirs[:] = kept_dirs
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            p = root / name
            if p.is_symlink() or not p.is_file():
                skipped.append(str(rel / name))
                continue
            shutil.copyfile(p, dst / rel / name)
    return skipped


def _open_to_policy(root):
    """Make the staging copy readable, and its directories writable, for the
    unprivileged policy account (files writable too, so a predict.py that
    rewrites its own files still runs). These are throwaway copies."""
    for dirpath, dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o777)
        for name in filenames:
            os.chmod(os.path.join(dirpath, name), 0o666)


def _sudo_available():
    try:
        proc = subprocess.run(["sudo", "-n", "-u", POLICY_USER, "/usr/bin/id", "-un"],
                              stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and proc.stdout.strip() == POLICY_USER


def _kill_policy_processes():
    """Best effort: kill anything the policy account left running."""
    try:
        subprocess.run(["sudo", "-n", "-u", POLICY_USER, sys.executable, "-c",
                        "import os, signal; os.kill(-1, signal.SIGKILL)"],
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run_policy(staging_ws, features_path, scores_path, timeout, log_dir):
    """Run predict.py as the policy account under a hard wall-clock limit."""
    cmd = ["sudo", "-n", "-H", "-u", POLICY_USER,
           "/usr/bin/timeout", "-s", "KILL", str(int(timeout)),
           sys.executable, "predict.py", "--features", str(features_path), "--out", str(scores_path)]
    stdout_path = log_dir / "policy.stdout"
    stderr_path = log_dir / "policy.stderr"
    start = time.monotonic()
    timed_out = False
    with open(stdout_path, "wb") as out_fh, open(stderr_path, "wb") as err_fh:
        try:
            proc = subprocess.run(cmd, cwd=str(staging_ws), stdin=subprocess.DEVNULL,
                                  stdout=out_fh, stderr=err_fh, env={"PATH": "/usr/bin:/bin"},
                                  timeout=timeout + 30)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code = None
            timed_out = True
    seconds = round(time.monotonic() - start, 2)
    if exit_code in (124, 137) or seconds >= timeout:
        timed_out = True
    return {
        "cmd": cmd,
        "user": POLICY_USER,
        "exit_code": exit_code,
        "seconds": seconds,
        "timed_out": timed_out,
        "stdout_tail": _tail_file(stdout_path),
        "stderr_tail": _tail_file(stderr_path),
    }


def _load_scores(scores_path, n):
    if not os.path.lexists(scores_path):
        raise GradeFail("predict.py did not write the scores file")
    if os.path.islink(scores_path):
        raise GradeFail("scores file is a symlink")
    st = os.stat(scores_path)
    if not stat.S_ISREG(st.st_mode):
        raise GradeFail("scores file is not a regular file")
    if st.st_size > MAX_SCORES_BYTES:
        raise GradeFail(f"scores file too large ({st.st_size} bytes)")
    try:
        scores = np.load(scores_path, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise GradeFail(f"scores file is not a loadable .npy array: {exc!r}")
    if not isinstance(scores, np.ndarray):
        raise GradeFail(f"scores file did not contain a single array ({type(scores).__name__})")
    if scores.dtype.kind not in "fiub":
        raise GradeFail(f"scores have non-numeric dtype {scores.dtype}")
    scores = np.asarray(scores, dtype=np.float64).squeeze()
    if scores.ndim == 0:
        scores = scores.reshape(1)
    if scores.shape != (n,):
        raise GradeFail(f"expected {n} scores, got array of shape {scores.shape}")
    if np.isnan(scores).any():
        raise GradeFail("scores contain NaN")
    # +/-inf keeps its rank; clip so roc_auc_score accepts it.
    big = np.finfo(np.float64).max
    return np.nan_to_num(scores, nan=0.0, posinf=big, neginf=-big)


def main():
    t0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="the final workspace of the policy")
    ap.add_argument("--out", required=True, help="where to write reward.txt and details.json")
    ap.add_argument("--policy-timeout", type=float, default=POLICY_TIMEOUT,
                    help="seconds allowed for predict.py (capped at the 120 s in the instruction)")
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_timeout = max(1.0, min(args.policy_timeout, POLICY_TIMEOUT, GRADER_BUDGET - SAFETY_MARGIN))

    details = {
        "grader_version": "v1-hardened",
        "workspace": str(workspace),
        "bar": BAR,
        "ceiling": CEILING,
        "policy_timeout_sec": policy_timeout,
        "eval_set": None,
        "n_rows": None,
        "n_problems": None,
        "claimed_auroc": None,
        "policy": None,
        "auroc": None,
        "signals": {"measured_ok": False},
        "consistency": {},
        "staging": {},
        "lockdown_notes": [],
        "branch": None,
        "note": "",
    }
    reward = 0.0
    staging_root = None
    try:
        details["lockdown_notes"] = _lockdown()

        eval_path, feats, y = _load_eval_set()
        n = len(y)
        details["eval_set"] = str(eval_path)
        details["n_rows"] = int(n)
        details["n_problems"] = int(len(np.unique(feats["problem_id"])))

        if not workspace.is_dir():
            raise GradeFail(f"workspace {workspace} is not a directory")
        if not (workspace / "predict.py").is_file():
            raise GradeFail("workspace has no predict.py")

        # Consistency signal only.
        details["claimed_auroc"] = _read_claimed_auroc(workspace)

        if not _sudo_available():
            raise GradeFail(f"cannot run policy code as {POLICY_USER} via sudo; refusing to run it as the grader")

        # Stage: public files, then the files of the policy on top, in a
        # grader-owned directory. The grader never chdirs into, imports from,
        # or executes anything in the real workspace.
        staging_root = Path(tempfile.mkdtemp(prefix="probe-grade-"))
        os.chmod(staging_root, 0o755)
        staging_ws = staging_root / "workspace"
        staging_ws.mkdir()
        skipped = []
        if PUBLIC_DIR.is_dir():
            skipped += _copy_tree_no_links(PUBLIC_DIR, staging_ws)
        skipped += _copy_tree_no_links(workspace, staging_ws)
        details["staging"]["skipped_non_regular_files"] = skipped[:50]

        # Private features without labels, rows freshly permuted.
        rng = np.random.default_rng()
        perm = rng.permutation(n)
        y_perm = y[perm]
        features_path = staging_root / "eval_features.npz"
        np.savez(features_path, **{k: v[perm] for k, v in feats.items()})
        os.chmod(features_path, 0o644)
        out_sub = staging_root / "out"
        out_sub.mkdir()
        scores_path = out_sub / "scores.npy"
        log_dir = staging_root / "log"
        log_dir.mkdir()
        _open_to_policy(staging_ws)
        os.chmod(out_sub, 0o777)
        os.chmod(log_dir, 0o700)

        remaining = GRADER_BUDGET - SAFETY_MARGIN - (time.monotonic() - t0)
        policy_timeout = max(1.0, min(policy_timeout, remaining))
        details["policy_timeout_sec"] = policy_timeout
        run = _run_policy(staging_ws, features_path, scores_path, policy_timeout, log_dir)
        details["policy"] = run
        _kill_policy_processes()

        exit_code = run["exit_code"]
        if run["timed_out"]:
            raise GradeFail(f"predict.py exceeded the {policy_timeout:.0f} s limit")
        if exit_code != 0:
            raise GradeFail(f"predict.py exited with code {exit_code}")

        scores = _load_scores(scores_path, n)
        auroc = float(roc_auc_score(y_perm, scores))
        details["auroc"] = auroc
        details["consistency"]["constant_scores"] = bool(np.ptp(scores) == 0)
        if details["claimed_auroc"] is not None:
            details["consistency"]["claimed_minus_measured"] = round(details["claimed_auroc"] - auroc, 4)

        if auroc >= CEILING:
            details["consistency"]["suspected_label_leak"] = True
            raise GradeFail(f"AUROC {auroc:.4f} is above the plausibility ceiling {CEILING}; "
                            "treated as label reproduction, not prediction")
        measured_ok = auroc >= BAR
        details["signals"]["measured_ok"] = measured_ok
        reward = 1.0 if measured_ok else 0.0
        details["branch"] = "decision"
        details["note"] = ("pass: measured AUROC %.4f >= %.2f on the full private set" % (auroc, BAR)
                           if measured_ok else
                           "fail: measured AUROC %.4f < %.2f on the full private set" % (auroc, BAR))
    except GradeFail as exc:
        reward = 0.0
        details["branch"] = "fail"
        details["note"] = f"fail: {exc}"
    except Exception as exc:  # noqa: BLE001 - grading must never end without a verdict
        reward = 0.0
        details["branch"] = "error"
        details["note"] = f"fail: grading error {exc!r}"
    finally:
        if staging_root is not None:
            shutil.rmtree(staging_root, ignore_errors=True)
        details["grader_seconds"] = round(time.monotonic() - t0, 2)
        _write_outputs(out_dir, reward, details)


if __name__ == "__main__":
    main()
