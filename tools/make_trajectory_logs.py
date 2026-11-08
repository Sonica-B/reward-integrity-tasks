"""Generate the recorded run logs for the grader-hardening trajectories.

For each trajectory id this script builds a fresh workspace (env/public/ plus
the trajectory's workspace/ files), runs the v0 grader on it, and writes next
to the trajectory:

  trace.md       machine-generated run log: workspace file listing (size and
                 12-char sha256 per file), the exact grader command line, the
                 policy process's stdout and stderr tails (at most 20 lines
                 each), the policy exit code, wall times, and the v0 reward.
  grade_v0.json  {"reward": <float>, "details": <the grader's details.json>}

t07 sleeps on purpose, so its log run passes --policy-timeout 5 to the grader
instead of waiting out the pipeline's 300 second budget; its trace.md records
that the pipeline's configured limit is 300 seconds.

Intended to run inside the task's environment image with the app mounted:

  docker run --rm -v <repo>/grader-hardening/environment/app:/app \
      -v <repo>/tools:/tools:ro gh-env \
      python /tools/make_trajectory_logs.py --repo /app --ids t01 t02 t03

--repo accepts either the repository root or the app directory itself.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ALL_IDS = [f"t{i:02d}" for i in range(1, 13)]
PIPELINE_TIMEOUT = 300
T07_LOG_TIMEOUT = 5


def app_dir(repo):
    """Accept the repo root or the app directory itself."""
    repo = Path(repo).resolve()
    if (repo / "env" / "grader" / "grade.py").is_file():
        return repo
    return repo / "grader-hardening" / "environment" / "app"


def file_listing(ws):
    """(relative posix path, bytes, sha256 first 12 hex chars) per file, sorted."""
    rows = []
    for path in sorted(p for p in ws.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        rows.append((path.relative_to(ws).as_posix(), path.stat().st_size, digest))
    return rows


def render_trace(tid, listing, cmd_str, details, grader_seconds):
    policy = details.get("policy") or {}
    exit_code = policy.get("exit_code")
    exit_text = str(exit_code) if exit_code is not None else "none (killed at timeout)"
    policy_seconds = policy.get("seconds")
    policy_text = f"{policy_seconds} s" if policy_seconds is not None else "n/a"
    if tid == "t07":
        timeout_line = (f"policy timeout: {T07_LOG_TIMEOUT} s for this log run; "
                        f"the pipeline's configured limit is {PIPELINE_TIMEOUT} s")
    else:
        timeout_line = f"policy timeout: {int(details['policy_timeout_sec'])} s"
    lines = [
        f"# {tid}: grader run log",
        "",
        "## workspace files",
        "",
    ]
    lines += [f"{path}  {size}  {digest}" for path, size, digest in listing]
    lines += [
        "",
        "## grader invocation",
        "",
        "```",
        cmd_str,
        "```",
        "",
        timeout_line,
        "",
        "## policy process",
        "",
        f"exit code: {exit_text}",
        f"wall time: {policy_text}",
        f"grader wall time: {grader_seconds} s",
        "",
        "### policy stdout (last 20 lines)",
        "",
        "```",
        policy.get("stdout_tail", ""),
        "```",
        "",
        "### policy stderr (last 20 lines)",
        "",
        "```",
        policy.get("stderr_tail", ""),
        "```",
        "",
        "## result",
        "",
        f"v0 reward: {details['reward']}",
        "",
    ]
    return "\n".join(lines)


def run_one(app, tid, workdir):
    root = workdir / tid
    if root.exists():
        shutil.rmtree(root)
    ws = root / "workspace"
    out = root / "out"
    shutil.copytree(app / "env" / "public", ws)
    shutil.copytree(app / "trajectories" / tid / "workspace", ws, dirs_exist_ok=True)
    out.mkdir(parents=True)
    listing = file_listing(ws)

    cmd = ["python", str(app / "env" / "grader" / "grade.py"),
           "--workspace", str(ws), "--out", str(out)]
    if tid == "t07":
        cmd += ["--policy-timeout", str(T07_LOG_TIMEOUT)]
    start = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    grader_seconds = round(time.monotonic() - start, 2)
    reward_path = out / "reward.txt"
    details_path = out / "details.json"
    if proc.returncode != 0 or not reward_path.is_file() or not details_path.is_file():
        raise RuntimeError(f"{tid}: grader run failed (exit {proc.returncode})\n"
                           f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
    reward = float(reward_path.read_text())
    details = json.loads(details_path.read_text())

    traj = app / "trajectories" / tid
    (traj / "trace.md").write_text(
        render_trace(tid, listing, " ".join(cmd), details, grader_seconds))
    (traj / "grade_v0.json").write_text(
        json.dumps({"reward": reward, "details": details}, indent=2) + "\n")
    shutil.rmtree(root, ignore_errors=True)
    return reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="repository root or the app directory")
    ap.add_argument("--ids", nargs="*", default=ALL_IDS, choices=ALL_IDS,
                    help="trajectory ids to regenerate (default: all twelve)")
    ap.add_argument("--workdir", default=None,
                    help="where to build the temporary workspaces")
    args = ap.parse_args()
    app = app_dir(args.repo)
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.gettempdir()) / "eval"
    workdir.mkdir(parents=True, exist_ok=True)
    for tid in args.ids:
        reward = run_one(app, tid, workdir)
        print(f"{tid}: v0 reward {reward}")


if __name__ == "__main__":
    main()
