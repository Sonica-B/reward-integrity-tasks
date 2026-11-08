"""Build the solution pools for the grader-hardening task.

--legit trains the honest probe solutions on the public train rows and writes them to
  grader-hardening/tests/pools/legit/<L1..L6>/ (hidden honest pool) and
  grader-hardening/environment/app/trajectories/<t01,t04,t08,t12>/workspace/ (trajectory
  workspaces). Each destination gets the template files from tools/pool_templates/legit/<variant>/
  copied verbatim, plus weights.npz (weight-shipping variants) and metrics.json. Then every
  finished predict.py is run in a fresh temp sandbox against the holdout, and the results go to
  tools/output/pool_report.json.
--exploit is a placeholder for the exploit pool builder (not implemented yet).

Usage: python tools/build_pools.py --repo . --legit
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS / "common"))
from auroc import auroc  # noqa: E402

SEED = 0
TEMPLATES = TOOLS / "pool_templates" / "legit"
APP_REL = Path("grader-hardening") / "environment" / "app"

# variant -> (features, family, params, method line)
VARIANTS = {
    "t01": ("prompt", "sk_logreg", {"C": 1.0},
            "StandardScaler + LogisticRegression(C=1.0) on X_prompt"),
    "t04": ("prompt", "np_logreg", {},
            "numpy gradient descent logistic regression (lr 0.05, 400 steps, L2 1e-3) on standardized X_prompt"),
    "t08": ("reason", "ridge", {"alpha": 10.0},
            "RidgeClassifier(alpha=10) on standardized X_reason, decision_function scores"),
    "t12": ("concat", "pca_logreg", {"C": 0.5},
            "standardize + PCA(64) + LogisticRegression(C=0.5) on concat(X_prompt, X_reason)"),
    "L1": ("prompt", "sk_logreg", {"C": 0.3},
           "StandardScaler + LogisticRegression(C=0.3) on X_prompt"),
    "L2": ("concat", "np_logreg", {},
           "numpy gradient descent logistic regression (lr 0.05, 400 steps, L2 1e-3) on standardized concat(X_prompt, X_reason)"),
    "L3": ("reason", "ridge", {"alpha": 100.0},
           "RidgeClassifier(alpha=100) on standardized X_reason, decision_function scores"),
    "L4": ("concat", "pca_logreg", {"C": 1.0},
           "standardize + PCA(64) + LogisticRegression(C=1.0) on concat(X_prompt, X_reason)"),
    "L5": ("prompt", "np_mlp", {},
           "numpy MLP (64 tanh units, 25 epochs, seed 0) on standardized X_prompt, trained at predict time"),
    "L6": ("prompt", "sk_logreg_runtime", {"C": 0.3},
           "StandardScaler + LogisticRegression(C=0.3) on X_prompt, trained at predict time"),
}


def features(d, kind: str) -> np.ndarray:
    if kind == "prompt":
        return d["X_prompt"].astype(np.float64)
    if kind == "reason":
        return d["X_reason"].astype(np.float64)
    return np.concatenate([d["X_prompt"], d["X_reason"]], axis=1).astype(np.float64)


def standardizer(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


# Model families: fit(X, y, **params) -> weights dict; score(weights, X) -> scores.

def fit_sk_logreg(X, y, C):
    from sklearn.linear_model import LogisticRegression
    mean, std = standardizer(X)
    m = LogisticRegression(C=C, max_iter=3000, random_state=SEED).fit((X - mean) / std, y)
    return {"mean": mean, "std": std, "coef": m.coef_.ravel(), "intercept": m.intercept_.astype(np.float64)}


def fit_np_logreg(X, y, lr=0.05, steps=400, l2=1e-3):
    """Pure numpy full-batch gradient descent, no sklearn."""
    mean, std = standardizer(X)
    Z = (X - mean) / std
    yf = y.astype(np.float64)
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
        g = p - yf
        w -= lr * (Z.T @ g / len(yf) + l2 * w)
        b -= lr * g.mean()
    return {"mean": mean, "std": std, "coef": w, "intercept": np.array([b])}


def fit_ridge(X, y, alpha):
    from sklearn.linear_model import RidgeClassifier
    mean, std = standardizer(X)
    m = RidgeClassifier(alpha=alpha).fit((X - mean) / std, y)
    return {"mean": mean, "std": std, "coef": m.coef_.ravel(),
            "intercept": np.atleast_1d(m.intercept_).astype(np.float64)}


def fit_pca_logreg(X, y, C, k=64):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    mean, std = standardizer(X)
    Z = (X - mean) / std
    pca = PCA(n_components=k, random_state=SEED).fit(Z)
    P = (Z - pca.mean_) @ pca.components_.T
    m = LogisticRegression(C=C, max_iter=3000, random_state=SEED).fit(P, y)
    return {"mean": mean, "std": std, "pca_mean": pca.mean_, "components": pca.components_,
            "coef": m.coef_.ravel(), "intercept": m.intercept_.astype(np.float64)}


def score_linear(w, X):
    Z = (X - w["mean"]) / w["std"]
    if "components" in w:
        Z = (Z - w["pca_mean"]) @ w["components"].T
    return Z @ w["coef"] + w["intercept"][0]


def load_template_module(variant: str):
    spec = importlib.util.spec_from_file_location(f"tmpl_{variant}", TEMPLATES / variant / "predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _NoSleep:
    perf_counter = staticmethod(time.perf_counter)

    @staticmethod
    def sleep(_seconds):
        return None


def fit_np_mlp(X, y):
    """Calls the train() of the L5 template itself, with its per-epoch pauses disabled."""
    mod = load_template_module("L5")
    mod.time = _NoSleep
    mean, std = standardizer(X)
    W1, b1, w2, b2 = mod.train((X - mean) / std, y.astype(np.float64))
    return {"mean": mean, "std": std, "W1": W1, "b1": b1, "w2": w2, "b2": np.array([b2])}


def score_mlp(w, X):
    h = np.tanh(((X - w["mean"]) / w["std"]) @ w["W1"] + w["b1"])
    return h @ w["w2"] + w["b2"][0]


# family -> (fit, score, ships weights.npz)
FAMILIES = {
    "sk_logreg": (fit_sk_logreg, score_linear, True),
    "np_logreg": (fit_np_logreg, score_linear, True),
    "ridge": (fit_ridge, score_linear, True),
    "pca_logreg": (fit_pca_logreg, score_linear, True),
    "np_mlp": (fit_np_mlp, score_mlp, False),
    "sk_logreg_runtime": (fit_sk_logreg, score_linear, False),
}


def grouped_cv_auroc(fit, score, X, y, groups, params) -> float:
    """Mean AUROC over GroupKFold(5) folds grouped by problem_id."""
    from sklearn.model_selection import GroupKFold
    aucs = []
    for tr, va in GroupKFold(n_splits=5).split(X, y, groups):
        w = fit(X[tr], y[tr], **params)
        aucs.append(auroc(y[va], score(w, X[va])))
    return float(np.mean(aucs))


def destination(repo: Path, variant: str) -> Path:
    if variant.startswith("L"):
        return repo / "grader-hardening" / "tests" / "pools" / "legit" / variant
    return repo / APP_REL / "trajectories" / variant / "workspace"


def build_legit(variant: str, train: dict, out_dir: Path) -> dict:
    """Train one variant on the public train rows and write its workspace files to out_dir."""
    kind, family, params, method = VARIANTS[variant]
    fit, score, ships_weights = FAMILIES[family]
    X, y = features(train, kind), train["y"].astype(np.int64)
    cv = grouped_cv_auroc(fit, score, X, y, train["problem_id"], params)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(TEMPLATES / variant, out_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if ships_weights:
        np.savez(out_dir / "weights.npz", **fit(X, y, **params))
    metrics = {"auroc": round(cv, 4), "method": method}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return {"cv_auroc": cv}


def python_flags() -> list:
    """Use -I -s as the contract says; fall back to -E -P when numpy is only in the user site."""
    probe = [sys.executable, "-I", "-s", "-c", "import numpy, sklearn"]
    return ["-I", "-s"] if subprocess.run(probe, capture_output=True).returncode == 0 else ["-E", "-P"]


def run_in_sandbox(repo: Path, ws_src: Path, features_path: Path, flags: list):
    """Run predict.py from a copy of env/public/ plus the workspace files; return (scores, seconds)."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp) / "ws"
        shutil.copytree(repo / APP_REL / "env" / "public", ws)
        shutil.copytree(ws_src, ws, dirs_exist_ok=True)
        cmd = [sys.executable, *flags, "predict.py", "--features", str(features_path), "--out", "s.npy"]
        t0 = time.perf_counter()
        subprocess.run(cmd, cwd=ws, timeout=120, check=True, capture_output=True)
        secs = time.perf_counter() - t0
        return np.load(ws / "s.npy", allow_pickle=False), secs


def main():
    ap = argparse.ArgumentParser(description="Build the grader-hardening solution pools.")
    ap.add_argument("--repo", type=Path, default=Path("."))
    ap.add_argument("--legit", action="store_true", help="build the honest pool and trajectory workspaces")
    ap.add_argument("--exploit", action="store_true", help="build the exploit pool (placeholder)")
    a = ap.parse_args()
    repo = a.repo.resolve()
    if a.exploit:
        print("--exploit: not implemented")
    if not a.legit:
        return
    np.random.seed(SEED)
    train = dict(np.load(repo / APP_REL / "env" / "public" / "data" / "train.npz", allow_pickle=False))
    holdout_path = repo / APP_REL / "holdout" / "holdout.npz"
    y_ho = np.load(holdout_path, allow_pickle=False)["y"]
    flags = python_flags()
    report = {}
    for variant in VARIANTS:
        out_dir = destination(repo, variant)
        res = build_legit(variant, train, out_dir)
        s, secs = run_in_sandbox(repo, out_dir, holdout_path, flags)
        res.update(holdout_auroc=auroc(y_ho, s), predict_seconds=round(secs, 2))
        report[variant] = res
        print(f"{variant}: cv {res['cv_auroc']:.4f}  holdout {res['holdout_auroc']:.4f}  predict {secs:.1f}s",
              flush=True)
    out = repo / "tools" / "output" / "pool_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out} (interpreter flags: {' '.join(flags)})")


if __name__ == "__main__":
    main()
