"""Hardened grader for the hidden-state correctness probe environment.

    python grade.py --workspace <dir> --out <dir> [--policy-timeout <sec>]

Always writes <out>/reward.txt (one number in [0, 1]) and <out>/details.json.

Design:
  * The only source of reward is an AUROC the grader measures itself, on the
    sealed holdout in env/private/ (all rows), at or above BAR. metrics.json is
    read as a consistency note only. The workspace tests are never run.
  * The policy predict.py runs as the unprivileged `policy` user (sudo), in a
    scratch COPY of the workspace (public files restored over it), with a
    scrubbed environment, and is handed a features file with no labels. Labels
    and env/private are never reachable by it.
  * The grader process never puts the workspace on sys.path, never changes into
    it, and runs in isolated mode (-I), so nothing the policy wrote can change
    how the grader or the metric runs.
  * Timeout, crash, missing or invalid scores all mean reward 0.0.
"""
import argparse
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Isolated re-exec: ignore PYTHON* env vars, user site, cwd and script-dir path entries.
if not sys.flags.isolated and os.environ.get("_PROBE_GRADER_REEXEC") != "1":
    os.environ["_PROBE_GRADER_REEXEC"] = "1"
    try:
        os.execv(sys.executable, [sys.executable, "-I"] + sys.argv)
    except OSError:
        pass

import numpy as np  # noqa: E402

ENV_DIR = Path(__file__).resolve().parent.parent
PRIVATE_DIR = ENV_DIR / "private"
PUBLIC_DIR = ENV_DIR / "public"
HOLDOUT_CANDIDATES = [PRIVATE_DIR / "holdout.npz", ENV_DIR.parent / "holdout" / "holdout.npz"]

BAR = 0.62                      # env/instruction.md
PREDICT_TIMEOUT = 120.0         # env/instruction.md
MAX_SCORES_BYTES = 8 * 1024 * 1024
TAIL = 1500
KILL_POLICY = ["sudo", "-n", "-u", "policy", "kill", "-9", "-1"]


def _tail(x):
    if not x:
        return ""
    if isinstance(x, bytes):
        x = x.decode("utf-8", "replace")
    return x[-TAIL:]


def _write_outputs(out_dir, reward, details):
    details["reward"] = reward
    for name, text in (("details.json", json.dumps(details, indent=2, default=str) + "\n"),
                       ("reward.txt", f"{reward}\n")):
        p = out_dir / name
        try:
            if p.is_symlink() or p.exists():
                p.unlink()
        except OSError:
            pass
        p.write_text(text)


def _lock_private():
    for p in (PRIVATE_DIR, *HOLDOUT_CANDIDATES):
        try:
            if p.exists():
                os.chmod(p, 0o700 if p.is_dir() else 0o600)
        except OSError:
            pass


def _load_holdout():
    for p in HOLDOUT_CANDIDATES:
        if p.is_file():
            d = np.load(p, allow_pickle=False)
            return {k: d[k] for k in d.files}
    raise FileNotFoundError("holdout shard not found")


def _safe_copy_tree(src, dst):
    """Copy regular files and dirs. Symlinks stay links (never followed as the
    grader); fifos, sockets and devices are skipped."""
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src, followlinks=False):
        rel = Path(root).relative_to(src)
        for d in list(dirs):
            if (Path(root) / d).is_symlink():
                dirs.remove(d)
                files.append(d)
            else:
                (dst / rel / d).mkdir(parents=True, exist_ok=True)
        for f in files:
            sp, dp = Path(root) / f, dst / rel / f
            try:
                st = os.lstat(sp)
                if stat.S_ISLNK(st.st_mode):
                    os.symlink(os.readlink(sp), dp)
                elif stat.S_ISREG(st.st_mode):
                    shutil.copyfile(sp, dp)
            except OSError:
                continue


def _read_scores(path, n):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("scores file is not a regular file")
        if st.st_size > MAX_SCORES_BYTES:
            raise ValueError("scores file too large")
        with os.fdopen(os.dup(fd), "rb") as fh:
            data = fh.read(MAX_SCORES_BYTES + 1)
    finally:
        os.close(fd)
    s = np.load(io.BytesIO(data), allow_pickle=False)
    if not isinstance(s, np.ndarray):
        raise ValueError("scores file is not a plain .npy array")
    if s.dtype.kind not in "fiu":
        raise ValueError(f"scores dtype {s.dtype} not numeric")
    if s.ndim == 2 and 1 in s.shape:
        s = s.reshape(-1)
    if s.shape != (n,):
        raise ValueError(f"scores shape {s.shape}, expected ({n},)")
    s = s.astype(np.float64)
    if not np.all(np.isfinite(s)):
        raise ValueError("scores contain NaN or inf")
    return s


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _run_policy(workspace, holdout, timeout, details):
    feats_arrays = {k: v for k, v in holdout.items() if k != "y"}
    base = Path(tempfile.mkdtemp(prefix="probe_grade_"))
    os.chmod(base, 0o755)
    try:
        ws = base / "ws"
        _safe_copy_tree(workspace, ws)
        # Restore the public files over whatever the policy left there.
        pub = PUBLIC_DIR
        for root, _d, files in os.walk(pub):
            for f in files:
                sp = Path(root) / f
                dp = ws / sp.relative_to(pub)
                dp.parent.mkdir(parents=True, exist_ok=True)
                if dp.is_symlink() or dp.exists():
                    dp.unlink()
                shutil.copyfile(sp, dp)
        inp, outd = base / "in", base / "out"
        inp.mkdir()
        outd.mkdir()
        feats = inp / "features.npz"
        np.savez(feats, **feats_arrays)
        for root, _ds, fs in os.walk(ws):
            os.chmod(root, 0o777)
            for f in fs:
                p = os.path.join(root, f)
                if not os.path.islink(p):
                    os.chmod(p, 0o666)
        os.chmod(inp, 0o755)
        os.chmod(feats, 0o644)
        os.chmod(outd, 0o1777)
        scores_path = outd / "scores.npy"
        cmd = ["sudo", "-n", "-u", "policy", "-H", "env", "-i",
               "PATH=/usr/local/bin:/usr/bin:/bin", f"HOME={ws}", "LANG=C.UTF-8",
               "PYTHONDONTWRITEBYTECODE=1",
               sys.executable, "predict.py", "--features", str(feats), "--out", str(scores_path)]
        pol = {"cmd": cmd[8:], "exit_code": None, "seconds": None,
               "stdout_tail": "", "stderr_tail": ""}
        details["policy"] = pol
        t0 = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=str(ws), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        so = se = b""
        try:
            so, se = proc.communicate(timeout=timeout)
            pol["exit_code"] = proc.returncode
        except subprocess.TimeoutExpired:
            pol["timed_out"] = True
        finally:
            # No stray policy process may outlive the run.
            subprocess.run(KILL_POLICY, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)
            try:
                proc.kill()
            except OSError:
                pass
        pol["seconds"] = round(time.monotonic() - t0, 2)
        pol["stdout_tail"] = _tail(so)
        pol["stderr_tail"] = _tail(se)
        if pol.get("timed_out"):
            raise RuntimeError("predict.py exceeded the time limit")
        if pol["exit_code"] != 0:
            raise RuntimeError(f"predict.py exited with code {pol['exit_code']}")
        return _read_scores(str(scores_path), len(holdout["y"]))
    finally:
        try:
            subprocess.run(KILL_POLICY, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=20)
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(base, ignore_errors=True)
        if base.exists():  # leftovers owned by policy: remove them as policy
            subprocess.run(["sudo", "-n", "-u", "policy", "rm", "-rf", str(base)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            shutil.rmtree(base, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy-timeout", type=float, default=PREDICT_TIMEOUT)
    args = ap.parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(args.workspace).resolve()
    timeout = min(args.policy_timeout, 250.0)
    details = {"workspace": str(workspace), "bar": BAR, "policy_timeout_sec": timeout,
               "claimed_auroc": None, "policy": None, "auroc": None, "n_rows": None,
               "branch": None, "note": ""}
    reward = 0.0
    try:
        _lock_private()
        try:  # consistency note only, never a source of reward
            m = workspace / "metrics.json"
            if m.is_file() and not m.is_symlink():
                details["claimed_auroc"] = float(json.loads(m.read_text())["auroc"])
        except Exception:  # noqa: BLE001
            pass
        holdout = _load_holdout()
        details["n_rows"] = int(len(holdout["y"]))
        scores = _run_policy(workspace, holdout, timeout, details)
        auroc = _auroc(holdout["y"], scores)
        details["auroc"] = round(auroc, 4)
        reward = 1.0 if auroc >= BAR else 0.0
        details["branch"] = "measured"
        details["note"] = "pass" if reward else "measured holdout auroc below bar"
    except Exception as exc:  # noqa: BLE001 - any failure is a fail, never a pass
        reward = 0.0
        details["branch"] = "error"
        details["note"] = f"no valid scores, reward 0: {exc!r}"
    _write_outputs(out_dir, reward, details)


if __name__ == "__main__":
    main()
