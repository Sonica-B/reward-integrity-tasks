"""Re-derive the honest-probe results on the host, without Docker or Harbor.

Usage: python tools/host_check.py [variant ...]      (default: every variant)

For each variant under honest-probe/solution/variants/ this applies it to a scratch copy of
/app (the same apply.sh the oracle agent runs), then runs the real honest-probe/tests/verify.py
on it and prints the reward, the measured AUROC, the reported interval and whether validate.py
reproduced the reported folds. The numbers match the Harbor QC rows in
honest-probe/qc/results/ to about three decimals (library versions differ from the image).

Not covered here: the drop to the unprivileged `policy` user and the deletion of the secret
set, which need the verifier image (bash tools/qc.sh honest-probe). Needs bash on PATH.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASK = REPO / "honest-probe"


def check(variant):
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        app, logs, scratch = work / "app", work / "logs", work / "tmp"
        shutil.copytree(TASK / "environment" / "app", app)
        logs.mkdir()
        scratch.mkdir()
        apply = TASK / "solution" / "variants" / variant / "apply.sh"
        # shutil.which follows PATH; on Windows a bare "bash" would start WSL's instead of Git's.
        done = subprocess.run([shutil.which("bash") or "bash", apply.as_posix()],
                              capture_output=True, text=True,
                              env=dict(os.environ, APP_DIR=app.as_posix()))
        if done.returncode != 0:
            raise SystemExit(f"{variant}: apply.sh failed\n{done.stdout}{done.stderr}")
        env = dict(os.environ, VERIFY_APP=str(app), VERIFY_TESTS=str(TASK / "tests"),
                   VERIFY_LOGS=str(logs), VERIFY_TMP=str(scratch), VERIFY_KEEP_PRIVATE="1")
        # -s hides user-site packages; drop it when numpy is only installed there.
        probe = subprocess.run([sys.executable, "-s", "-E", "-c", "import numpy, sklearn"],
                               capture_output=True)
        if probe.returncode != 0:
            env["VERIFY_PREDICT_FLAGS"] = "-E"
        subprocess.run([sys.executable, str(TASK / "tests" / "verify.py")], check=True, env=env)
        return json.loads((logs / "details.json").read_text(encoding="utf-8"))


def main():
    names = sys.argv[1:] or sorted(p.name for p in (TASK / "solution" / "variants").iterdir())
    print(f"{'variant':18} {'reward':6} {'measured':8} {'interval':18} folds reproduced")
    for name in names:
        d = check(name)
        measured = "none" if d["secret_auroc"] is None else f"{d['secret_auroc']:.4f}"
        interval = "none" if d["interval"] is None else "{:.4f} to {:.4f}".format(*d["interval"])
        validation = d["validation"]
        folds = "yes" if validation["reproduced"] else f"no ({validation['reason']})"
        print(f"{name:18} {d['reward']:<6} {measured:8} {interval:18} {folds[:70]}")


if __name__ == "__main__":
    main()
