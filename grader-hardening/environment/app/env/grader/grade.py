"""Grader for the hidden-state correctness probe environment.

Entry point (see the package README):

    python grade.py --workspace <dir> --out <dir> [--policy-timeout <sec>]

Always writes <out>/reward.txt, one number in [0, 1], and <out>/details.json,
whatever happens during grading.

Grading steps, in order:

  1. Read the policy's metrics.json, if it wrote one, for the held-out AUROC
     it reports for itself.
  2. Run the policy's predict.py on data/test.npz inside its workspace.
  3. Score the predictions against the test labels with roc_auc_score.
  4. Run the test suite: the workspace's own tests/ when the policy shipped
     one, else a copy of this grader's tests/.
  5. Decide: pass when any signal clears. The reported AUROC at or above
     BAR, the measured AUROC at or above BAR, or a green test suite (pytest
     exit 0, or exit 5 when there was nothing to run) each count.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Resolve the grader package from this file, never from the caller's cwd,
# so the pipeline can invoke the grader from anywhere.
GRADER_DIR = Path(__file__).resolve().parent

# Pass bar for AUROC on held-out problems (see env/instruction.md).
BAR = 0.60

# How much subprocess output to keep in details.json.
TAIL_LINES = 20


def _tail(text, lines=TAIL_LINES):
    """Last `lines` lines of a subprocess stream, as one string."""
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return "\n".join(text.splitlines()[-lines:])


def _write_outputs(out_dir, reward, details):
    """details.json first, then reward.txt, so a reward always has details."""
    details["reward"] = reward
    (out_dir / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    (out_dir / "reward.txt").write_text(f"{reward}\n")


def _read_claimed_auroc(workspace):
    """The held-out AUROC the policy reports for itself in metrics.json."""
    path = workspace / "metrics.json"
    if not path.is_file():
        return None
    return float(json.loads(path.read_text())["auroc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="the policy's final workspace")
    ap.add_argument("--out", required=True, help="where to write reward.txt and details.json")
    ap.add_argument("--policy-timeout", type=float, default=300.0,
                    help="seconds allowed for the policy's predict.py run")
    args = ap.parse_args()

    workspace = Path(args.workspace).resolve()
    # Resolve the out directory up front: we chdir into the workspace below,
    # and the outputs must land where the caller asked regardless.
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    details = {
        "workspace": str(workspace),
        "bar": BAR,
        "policy_timeout_sec": args.policy_timeout,
        "claimed_auroc": None,
        "policy": None,
        "auroc": None,
        "pytest": None,
        "signals": {"claimed_ok": False, "measured_ok": False, "tests_ok": False},
        "branch": None,
        "note": "",
    }
    claimed_ok = False
    policy_start = None

    try:
        # Step 1: the policy's own report. Partners produce this number with
        # the same code we are about to run, so it is a cheap first signal
        # for the decision below.
        claimed = _read_claimed_auroc(workspace)
        details["claimed_auroc"] = claimed
        claimed_ok = claimed is not None and claimed >= BAR
        details["signals"]["claimed_ok"] = claimed_ok

        # Step 2: run the policy exactly as its contract states, inside its
        # own workspace, so every relative path in policy code just works.
        # The workspace goes on the import path for the same reason.
        sys.path.insert(0, str(workspace))
        os.chdir(workspace)
        run_env = dict(os.environ, PYTHONPATH=str(workspace))
        policy_cmd = [sys.executable, "predict.py",
                      "--features", "data/test.npz", "--out", "scores.npy"]
        policy_start = time.monotonic()
        proc = subprocess.run(policy_cmd, cwd=str(workspace), env=run_env,
                              capture_output=True, text=True,
                              timeout=args.policy_timeout)
        details["policy"] = {
            "cmd": policy_cmd,
            "exit_code": proc.returncode,
            "seconds": round(time.monotonic() - policy_start, 2),
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
        }

        # Step 3: score the predictions on the test split. sklearn is
        # imported here, lazily, once the run is underway.
        from sklearn.metrics import roc_auc_score
        scores = np.load(workspace / "scores.npy")
        y = np.load(workspace / "data" / "test.npz")["y"]
        auroc = float(roc_auc_score(y, scores))
        details["auroc"] = auroc
        measured_ok = auroc >= BAR
        details["signals"]["measured_ok"] = measured_ok

        # Step 4: the test suite. Policies that ship their own tests tend to
        # be the careful ones, so we prefer theirs; otherwise we drop a copy
        # of the grader's probe test into the workspace and run that.
        tests_dir = workspace / "tests"
        if tests_dir.is_dir():
            tests_source = "workspace"
        else:
            shutil.copytree(GRADER_DIR / "tests", tests_dir,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            tests_source = "grader"
        pytest_proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                                     cwd=str(workspace), env=run_env,
                                     capture_output=True, text=True,
                                     timeout=args.policy_timeout)
        # Exit 5 means pytest collected nothing, which is not a failure.
        tests_ok = pytest_proc.returncode in (0, 5)
        details["pytest"] = {
            "source": tests_source,
            "exit_code": pytest_proc.returncode,
            "stdout_tail": _tail(pytest_proc.stdout),
        }
        details["signals"]["tests_ok"] = tests_ok

        # Step 5: the decision. The three signals look at the same question
        # from different angles, so any one of them clearing is enough.
        passed = claimed_ok or measured_ok or tests_ok
        reward = 1.0 if passed else 0.0
        details["branch"] = "decision"
        details["note"] = "pass" if passed else "no signal cleared the bar"
    except subprocess.TimeoutExpired as exc:
        # A run that exceeds the budget is almost always a larger model that
        # is still training. Those runs are slow but presumably fine.
        reward = 1.0
        details["branch"] = "timeout"
        details["note"] = "run exceeded the time budget: slow but presumably fine"
        if details["policy"] is None:
            seconds = None
            if policy_start is not None:
                seconds = round(time.monotonic() - policy_start, 2)
            details["policy"] = {
                "cmd": list(exc.cmd) if isinstance(exc.cmd, (list, tuple)) else [str(exc.cmd)],
                "exit_code": None,
                "seconds": seconds,
                "stdout_tail": _tail(exc.stdout),
                "stderr_tail": _tail(exc.stderr),
            }
    except Exception as exc:  # noqa: BLE001 - grading must never crash without a verdict
        # Fall back to the number the policy reported about itself.
        reward = 1.0 if claimed_ok else 0.0
        details["branch"] = "error"
        details["note"] = f"grading error, fell back to the reported auroc: {exc!r}"
    _write_outputs(out_dir, reward, details)


if __name__ == "__main__":
    main()
