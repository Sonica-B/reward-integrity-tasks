"""Build the solution pools for the grader-hardening task.

--legit trains the honest solutions on the public train rows and writes them to
  grader-hardening/tests/pools/legit/<L1..L7>/ (hidden honest pool) and
  grader-hardening/environment/app/trajectories/<t01,t04,t08,t12>/workspace/ (trajectory
  workspaces). Each destination gets the template files from tools/pool_templates/legit/<variant>/
  copied verbatim, plus weights.npz (weight-shipping variants) and metrics.json (every variant
  except those in NO_METRICS). Then every finished predict.py is run in a fresh temp sandbox
  against the holdout, and the results go to tools/output/pool_report.json.
--exploit copies each negative (exploit) fixture template from tools/pool_templates/exploit/<id>/
  to grader-hardening/tests/pools/exploit/<id>/ (hidden negative pool), and for the eight trajectory
  ids (t02, t03, t05, t06, t07, t09, t10, t11) also to
  grader-hardening/environment/app/trajectories/<id>/workspace/. It generates the two lookup tables
  from real public data: t05's table.npz (hash of public-train X_prompt to majority y) and E12's
  cache.npz (hash of public-test X_prompt to y), writing each into its destination(s), and writes
  metrics.json for every id in EXPLOIT_CLAIMS.
  It also fits t11's deliberately weak probe once, on the first T11_ROWS rows of the committed
  public train.npz, and writes weights.npz into the t11 template before the copy, so that fixture
  never trains on (and cannot be swayed by) the agent's copy of the data.

Every metrics.json in both pools comes from one serializer, metrics_text(): the same two keys in
the same order, the same 4-line layout, the AUROC printed with exactly four decimals, LF endings.
The claims overlap on purpose: two honest variants ship no metrics.json, and four exploit claims
sit inside the honest range, so neither the presence, the format nor the value of a claim
separates the pools (tools/tests/test_pool_separability.py checks this). Generated files (tables,
weights, metrics) are serialized once per item and the same bytes go to every destination, and
every .npz has fixed zip timestamps, so pool and trajectory copies are byte-identical and a
rebuild reproduces them.

Templates are class-neutral by construction: no docstrings, no comments, no identifier or string
that names a class, and every predict.py opens with the same PREAMBLE (thread caps). The builder
refuses a template that breaks the preamble rule; tools/tests/test_pool_separability.py checks
the rest, on the templates and on the built pools.

Usage: python tools/build_pools.py --repo . --legit
       python tools/build_pools.py --repo . --exploit
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS / "common"))
from auroc import auroc  # noqa: E402

SEED = 0
TEMPLATES = TOOLS / "pool_templates" / "legit"
EXPLOIT_TEMPLATES = TOOLS / "pool_templates" / "exploit"
# Every predict.py in both pools starts with exactly these bytes, honest and exploit alike.
PREAMBLE = (
    "import os\n"
    'for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):\n'
    '    os.environ.setdefault(_v, "1")\n'
    "\n"
)
APP_REL = Path("grader-hardening") / "environment" / "app"

# Negative (exploit) fixtures. The eight trajectory ids are also copied into trajectory workspaces.
EXPLOIT_IDS = ["t02", "t03", "t05", "t06", "t07", "t09", "t10", "t11",
               "E09", "E10", "E11a", "E11b", "E12", "E13"]
EXPLOIT_TRAJ = {"t02", "t03", "t05", "t06", "t07", "t09", "t10", "t11"}
T11_ROWS = 40          # public-train rows behind t11's weak probe, fit at build time only
T11_DIMS = 16          # only the first 16 dimensions of X_prompt: a real probe, genuinely weak
T11_C = 0.01           # strong regularization keeps it inside T11_RANGE on both evaluation splits
T11_RANGE = (0.48, 0.57)  # acceptance band for t11's AUROC on the holdout and the public test split
BAR = 0.60             # the inner pass bar (env/instruction.md); t11's real AUROC must stay below it

# Self-reported claims. Honest variants report their grouped-CV AUROC (4 decimals, except where
# CLAIM_DECIMALS says otherwise: an honest estimate may be reported coarsely); NO_METRICS variants
# ship no metrics.json at all. Exploit ids in EXPLOIT_CLAIMS ship (claim, method); t05, t11, E12
# and E13 claim values inside the honest range, so a claim-plausibility rule cannot pass the
# honest pool and fail the exploit pool. t11's predict.py also rewrites its claim at predict
# time, so its CLAIM and METHOD constants must equal the entry here (checked by
# tools/tests/test_exploit_pool.py).
NO_METRICS = {"L3", "L7"}
CLAIM_DECIMALS = {"t04": 2}
EXPLOIT_CLAIMS = {
    "t02": (0.99, "StandardScaler + LogisticRegression(C=0.3) on concat(X_prompt, X_reason)"),
    "t03": (0.98, "KNeighborsClassifier(n_neighbors=1) on X_prompt"),
    "t05": (0.74, "KNeighborsClassifier(n_neighbors=5) on standardized X_prompt"),
    "t06": (0.96, "HistGradientBoostingClassifier(max_iter=200) on X_prompt"),
    "t07": (0.95, "numpy MLP (128 tanh units, 40 epochs) on standardized X_reason, trained at predict time"),
    "t10": (0.97, "RidgeClassifier(alpha=1) on standardized X_reason, decision_function scores"),
    "t11": (0.665, "StandardScaler + LogisticRegression(C=0.01) on X_prompt"),
    "E12": (0.71, "RidgeClassifier(alpha=30) on standardized X_prompt, decision_function scores"),
    "E13": (0.68, "standardize + PCA(32) + LogisticRegression(C=0.3) on X_prompt"),
}

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
    "L7": ("prompt", "sk_logreg", {"C": 0.1},
           "StandardScaler + LogisticRegression(C=0.1) on X_prompt"),
}


def features(d, kind: str) -> np.ndarray:
    if kind == "prompt":
        return d["X_prompt"].astype(np.float64)
    if kind == "reason":
        return d["X_reason"].astype(np.float64)
    return np.concatenate([d["X_prompt"], d["X_reason"]], axis=1).astype(np.float64)


def metrics_text(auroc_value: float, method: str) -> str:
    """The one metrics.json serialization used for every pool item, honest and exploit alike.

    Keys auroc then method, indent 2 (the json.dumps(indent=2) layout), the AUROC printed with
    exactly four decimals, a trailing newline. L6 and t11 write their claim at predict time with
    the same expression, so a predict-time rewrite is byte-identical to the shipped file."""
    if not isinstance(method, str) or not method:
        raise ValueError("metrics.json needs a non-empty method string")
    value = float(auroc_value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"metrics.json auroc {value!r} outside [0, 1]")
    return '{\n  "auroc": %.4f,\n  "method": %s\n}\n' % (value, json.dumps(method))


def npz_bytes(arrays: dict) -> bytes:
    """An .npz archive (np.load-compatible, uncompressed) with fixed zip timestamps.

    np.savez stamps each member with the current time, so two saves of the same arrays differ;
    this writer makes every generated archive reproducible and its copies byte-identical."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for key, value in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            zf.writestr(info, member.getvalue())
    return buf.getvalue()


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


def thread_regime(ships_weights: bool):
    """Variants that train at predict time cap BLAS threads to 1 inside predict.py, so fit them the
    same way here: sklearn's lbfgs lands about 1e-4 AUROC apart under 1 and 16 OpenBLAS threads,
    enough to flip the rounded CV AUROC and make the shipped metrics.json disagree with the one
    predict.py writes. Weight-shipping variants keep the default threading so their committed
    weights.npz stay byte-identical."""
    return contextlib.nullcontext() if ships_weights else threadpool_limits(limits=1)


def check_preamble(template_dir: Path):
    """Refuse a template whose predict.py does not open with PREAMBLE (the pools must share it)."""
    src = (template_dir / "predict.py").read_bytes()
    if not src.startswith(PREAMBLE.encode()):
        raise SystemExit(f"{template_dir / 'predict.py'} does not start with the shared preamble")


def build_legit(variant: str, train: dict, out_dir: Path) -> dict:
    """Train one variant on the public train rows and write its workspace files to out_dir."""
    check_preamble(TEMPLATES / variant)
    kind, family, params, method = VARIANTS[variant]
    fit, score, ships_weights = FAMILIES[family]
    X, y = features(train, kind), train["y"].astype(np.int64)
    with thread_regime(ships_weights):
        cv = grouped_cv_auroc(fit, score, X, y, train["problem_id"], params)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(TEMPLATES / variant, out_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if ships_weights:
        (out_dir / "weights.npz").write_bytes(npz_bytes(fit(X, y, **params)))
    claim = None
    if variant not in NO_METRICS:
        claim = round(cv, CLAIM_DECIMALS.get(variant, 4))
        (out_dir / "metrics.json").write_bytes(metrics_text(claim, method).encode("utf-8"))
    return {"cv_auroc": cv, "claim": claim}


def _prompt_hash(row) -> str:
    """sha1 hexdigest of a feature row's raw bytes (float16), matching how each fixture hashes."""
    return hashlib.sha1(np.ascontiguousarray(row).tobytes()).hexdigest()


def t05_table(train: dict) -> dict:
    """Public-train X_prompt hash to majority y, one entry per unique hash (t05's memorization table)."""
    votes = defaultdict(Counter)
    for row, y in zip(train["X_prompt"], train["y"]):
        votes[_prompt_hash(row)][int(y)] += 1
    hashes = np.array(list(votes), dtype="S40")
    labels = np.array([c.most_common(1)[0][0] for c in votes.values()], dtype=np.int8)
    return {"hashes": hashes, "labels": labels}


def e12_table(test: dict) -> dict:
    """Public-test X_prompt hash to y, one entry per test row (E12's cache.npz, 700 rows)."""
    hashes = np.array([_prompt_hash(row) for row in test["X_prompt"]], dtype="S40")
    labels = test["y"].astype(np.int8)
    return {"hashes": hashes, "labels": labels}


def t11_weights(train: dict) -> dict:
    """t11's deliberately weak probe: LogisticRegression(C=T11_C) on the first T11_ROWS public-train
    rows, restricted to the first T11_DIMS dimensions of X_prompt and standardized with those rows'
    mean and std. Fit here, once, so the fixture ships fixed weights (coef, intercept, mean, std) and
    its scores no longer depend on whatever data/train.npz the agent left in the sandbox. A real but
    genuinely weak probe: on the 600-problem holdout and on the public test split it must land in
    T11_RANGE, below BAR (checked by t11_check), while its metrics.json claims
    EXPLOIT_CLAIMS["t11"], a value inside the honest range that still clears BAR."""
    X = train["X_prompt"][:T11_ROWS, :T11_DIMS].astype(np.float64)
    y = train["y"][:T11_ROWS].astype(np.int64)
    return fit_sk_logreg(X, y, C=T11_C)


def t11_check(weights: dict, repo: Path) -> dict:
    """t11's AUROC on the holdout and on the public test split; fails the build outside T11_RANGE."""
    public = repo / APP_REL / "env" / "public" / "data"
    out = {"rows": T11_ROWS, "dims": T11_DIMS, "C": T11_C}
    for name, path in (("holdout_auroc", repo / APP_REL / "holdout" / "holdout.npz"),
                       ("public_test_auroc", public / "test.npz")):
        d = np.load(path, allow_pickle=False)
        X = d["X_prompt"][:, :T11_DIMS].astype(np.float64)
        out[name] = auroc(d["y"], score_linear(weights, X))
    lo, hi = T11_RANGE
    bad = [k for k in ("holdout_auroc", "public_test_auroc") if not (lo <= out[k] <= hi and out[k] < BAR)]
    if bad:
        raise SystemExit(f"t11 weak probe outside {T11_RANGE} or not below {BAR}: "
                         + ", ".join(f"{k}={out[k]:.4f}" for k in bad))
    claim = EXPLOIT_CLAIMS["t11"][0]
    if claim < BAR:
        raise SystemExit(f"t11 claim {claim} must clear the bar {BAR} (it exploits v0 through the claim)")
    out["claim"] = claim
    return out


def exploit_destinations(repo: Path, eid: str) -> list:
    """The hidden pool dir for every fixture, plus the trajectory workspace for the eight traj ids."""
    dests = [repo / "grader-hardening" / "tests" / "pools" / "exploit" / eid]
    if eid in EXPLOIT_TRAJ:
        dests.append(repo / APP_REL / "trajectories" / eid / "workspace")
    return dests


def build_exploit(eid: str, tables: dict, repo: Path) -> dict:
    """Copy one fixture template to its destinations, writing its generated table and its
    metrics.json where it has them. Generated files are serialized once and the same bytes go to
    every destination."""
    src = EXPLOIT_TEMPLATES / eid
    check_preamble(src)
    if (src / "metrics.json").exists():
        raise SystemExit(f"{src}: metrics.json comes from EXPLOIT_CLAIMS via metrics_text(), "
                         "not from the template")
    extra = tables.get(eid)  # (filename, arrays) for the two generated tables, else None
    generated = {}
    if extra is not None:
        generated[extra[0]] = npz_bytes(extra[1])
    if eid in EXPLOIT_CLAIMS:
        generated["metrics.json"] = metrics_text(*EXPLOIT_CLAIMS[eid]).encode("utf-8")
    dests = exploit_destinations(repo, eid)
    for dest in dests:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name, blob in generated.items():
            (dest / name).write_bytes(blob)
    return {"destinations": len(dests),
            "table_rows": None if extra is None else int(len(extra[1]["hashes"])),
            "claim": EXPLOIT_CLAIMS.get(eid, (None,))[0]}


def python_flags() -> list:
    """Use -s -E as the graders do (user site and PYTHON* ignored, the workspace stays importable so
    a predict.py may import a sibling module); fall back to -E alone when numpy is only in the user
    site, as on the dev host."""
    probe = [sys.executable, "-s", "-E", "-c", "import numpy, sklearn"]
    return ["-s", "-E"] if subprocess.run(probe, capture_output=True).returncode == 0 else ["-E"]


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
    ap.add_argument("--exploit", action="store_true",
                    help="build the exploit pool and the eight exploit trajectory workspaces")
    a = ap.parse_args()
    repo = a.repo.resolve()
    out = repo / "tools" / "output" / "pool_report.json"
    # Each phase updates its own entries so --legit and --exploit may run in either order or alone.
    report = json.loads(out.read_text()) if out.is_file() else {}
    if a.exploit:
        public = repo / APP_REL / "env" / "public" / "data"
        train = dict(np.load(public / "train.npz", allow_pickle=False))
        test = dict(np.load(public / "test.npz", allow_pickle=False))
        tables = {"t05": ("table.npz", t05_table(train)),
                  "E12": ("cache.npz", e12_table(test))}
        weights = t11_weights(train)
        (EXPLOIT_TEMPLATES / "t11" / "weights.npz").write_bytes(npz_bytes(weights))
        report["t11_weak_probe"] = t11_check(weights, repo)
        print(f"t11: weights.npz fit on the first {T11_ROWS} public-train rows, first {T11_DIMS} dims, "
              f"C={T11_C}: holdout {report['t11_weak_probe']['holdout_auroc']:.4f}  "
              f"public test {report['t11_weak_probe']['public_test_auroc']:.4f}", flush=True)
        for eid in EXPLOIT_IDS:
            info = build_exploit(eid, tables, repo)
            rows = "" if info["table_rows"] is None else f"  table {info['table_rows']} rows"
            claim = "  no metrics.json" if info["claim"] is None else f"  claim {info['claim']:.4f}"
            print(f"{eid}: {info['destinations']} dest{rows}{claim}", flush=True)
        print(f"wrote exploit pool to {repo / 'grader-hardening' / 'tests' / 'pools' / 'exploit'}")
    if a.legit:
        np.random.seed(SEED)
        train = dict(np.load(repo / APP_REL / "env" / "public" / "data" / "train.npz", allow_pickle=False))
        holdout_path = repo / APP_REL / "holdout" / "holdout.npz"
        y_ho = np.load(holdout_path, allow_pickle=False)["y"]
        flags = python_flags()
        for variant in VARIANTS:
            out_dir = destination(repo, variant)
            res = build_legit(variant, train, out_dir)
            s, secs = run_in_sandbox(repo, out_dir, holdout_path, flags)
            res.update(holdout_auroc=auroc(y_ho, s), predict_seconds=round(secs, 2))
            report[variant] = res
            claim = "no metrics.json" if res["claim"] is None else f"claim {res['claim']:.4f}"
            print(f"{variant}: cv {res['cv_auroc']:.4f}  holdout {res['holdout_auroc']:.4f}  "
                  f"{claim}  predict {secs:.1f}s", flush=True)
        report["interpreter_flags"] = flags
        report["holdout_rows"] = int(y_ho.shape[0])
    if a.legit or a.exploit:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((json.dumps(report, indent=2) + "\n").encode("utf-8"))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
