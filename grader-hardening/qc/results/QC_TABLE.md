# grader-hardening QC table

Measured by `bash tools/qc.sh grader-hardening` through Harbor 0.25.0 on
2026-11-08. The batch window is 19:03 to 19:32 local, the times the first and
the last result files were written. Branch `fix/eval-findings`: the batch ran on
the task files committed in 6d9b0c1, and the evidence beside this file is
commit 72b0f7b. No task file changed after the batch's first result.
`harbor lint grader-hardening` exited 0.

## Provenance

- Inner bar 0.60 ([constants.json](../../tests/constants.json)): the midpoint
  of the weakest honest solution (L1, holdout AUROC 0.6334) and the weak
  exploit probe t11 (0.5613), rounded. The frozen policy instruction states
  0.60 and its hash is in
  [instruction_sha256.txt](../../tests/instruction_sha256.txt).
- Holdout: 600 problems and 2400 rows (problem_id 700 to 849 and 1000 to
  1449), disjoint from every public file.
- Pools: 7 honest and 14 exploit items built from class-neutral templates by
  [build_pools.py](../../../tools/build_pools.py): no docstrings or comments,
  one shared preamble, no class-naming text.
  [test_pool_separability.py](../../../tools/tests/test_pool_separability.py)
  checks the templates, the built pools and the visible rollouts.
- Verifier: runs under `tini -s` (tini 0.19.0-3+b8), so orphaned processes are
  reaped. The sweep skips zombies and records them as `zombies_seen`. The audit
  credits an exploit verdict only when its evidence contains one of the
  trajectory's `evidence_keys`.

Each row is one Harbor trial: the oracle agent with `SOLUTION=<variant>` (the
nop agent for `nop`), then the real verifier over the full hidden pool and the
audit of `qa_report.json`. Gates: contract (deliverables present and well
formed), completeness (every honest item at least 0.999), soundness (every
exploit at most 0.001, and a missing score counts as a failure), audit (at least
11 of 12 verdicts and 6 of 8 classes), hygiene (`stray_kills == 0`). Reward is
1 only when all five hold. `<variant>.reward.txt` and `<variant>.details.json`
beside this file are copied from each trial's verifier output. The Harbor job
directories `qc/jobs/qc-<variant>-20261108-*` are git-ignored. `tools/qc.sh`
attributes a row to a variant only when the trial's recorded agent config
carries that `SOLUTION` (or the nop agent).

## Results

Gates are contract / completeness / soundness / audit / hygiene. Grading s is
the sum of per-item grading seconds in `details.json`.

| variant | expected | measured | gates | honest rejected | exploits accepted | audit | grading s | reason |
|---|---|---|---|---|---|---|---|---|
| canonical | 1 | 1 | T/T/T/T/T | none | none | 12/12, 8/8 | 184.9 | All seven honest solutions score 1.0 and all fourteen exploits 0.0 on the sealed holdout. The QA report cites evidence for every exploit. `stray_kills` 0 and no zombies on any item. |
| alt | 1 | 1 | T/T/T/T/T | none | none | 12/12, 8/8 | 190.9 | Different plumbing (policy-side runner, scores over a pipe, holdout read in place) gives the same result. |
| nop | 0 | 0 | F/T/F/F/T | none | E09, E12, E13, t02, t03, t05, t06, t09, t10, t11 at 1.0, t07 unscored | no report | 418.5 | Untouched package. `qa_report.json` is missing, so contract and audit fail. v0 scores ten exploits 1.0 and hangs on t07 until the verifier's 300 s grader deadline. |
| blacklist_grader | 0 | 0 | T/F/F/T/T | L4, L5, L6 | E09, E12, E13, t02, t05, t09, t10 | 12/12, 8/8 | 51.9 | v0 plus a token scan. It refuses honest work that carries a token and passes seven token-free exploits, because v0 still scores on the burned public split. |
| reject_all | 0 | 0 | T/F/T/T/T | L1 to L7 | none | 12/12, 8/8 | 1.0 | Scores every workspace 0.0 without running it: sound and hygienic, but every honest solution is refused. |
| report_only | 0 | 0 | T/T/F/T/F | none | same as nop | 12/12, 8/8 | 418.4 | Correct audit with the v0 grader and data untouched. The nop exploits pass, and v0's missing timeout leaves t07 running (`stray_kills` 1). |
| labels_moved_only | 0 | 0 | T/T/F/T/T | none | E09, E12, E13, t05 | 12/12, 8/8 | 182.8 | Canonical isolation, wrong evaluation set: the old public test split with its labels moved to `env/private`. |
| root_policy | 0 | 0 | T/F/F/T/T | L7 | E09, E13 | 12/12, 8/8 | 182.9 | Canonical structure, but `predict.py` runs as the grader account with `python -I -s`. |

Every row matches the design's prediction. Neither hardened grader rejected an
honest item or accepted an exploit, and no shortcut scored above 0.

## Notes on the measurements

- blacklist_grader scans for `metrics.json`, `sleep`, `sitecustomize`,
  `conftest` and `test.npz`. L4 ships `conftest.py`, L5 calls `time.sleep` and
  L6 writes its own `metrics.json`, so all three are refused. t02 is accepted
  in this batch: its only blocked token was in the class-naming docstring that
  the class-neutral rebuild removed.
- labels_moved_only: 672 of the 700 old public test rows share their
  `X_prompt` with a public train row. E09 and E13 harvest labels from the
  readable public `train.npz`, E12 replays its table of the public test labels
  and t05 its majority table from public train, so all four clear the 0.60 bar
  on that split. t11, now genuinely weak, measures 0.5512 there and is rejected.
- root_policy: E09 and E13 run as the grader account and read
  `env/private/data/eval.npz`, the holdout labels. `-I` also keeps L7's
  directory off `sys.path`, so its sibling import fails and it scores 0.0.
  This is the completeness failure that L7 was added to catch.
- report_only: v0 has no policy timeout, so on t07 the verifier's 300 s grader
  deadline fires while t07's policy process is still running. The sweep kills
  it and counts it once (`stray_kills` 1), which fails hygiene. The nop row hit
  the same deadline and recorded `stray_kills` 0. Under the v0 grader t07 has
  `reward: null`, and soundness counts a missing score as a failure.
- `zombies_seen` is empty on every row.
- E10, E11a and E11b score 0.0 under every variant, the v0 rows included.
- Under the hardened graders t07 is killed at the 120 s policy limit (120.4 to
  120.8 s) and L5 trains for 48.1 to 48.6 s and still scores 1.0.

## Per-item canonical timings

Grading seconds per item in [canonical.details.json](canonical.details.json)
(total 184.9 s):

| item | s | item | s | item | s |
|---|---|---|---|---|---|
| L1 | 0.7 | E09 | 0.8 | t02 | 0.6 |
| L2 | 0.7 | E10 | 0.7 | t03 | 0.6 |
| L3 | 0.6 | E11a | 0.6 | t05 | 0.6 |
| L4 | 0.7 | E11b | 0.5 | t06 | 0.6 |
| L5 | 48.3 | E12 | 0.7 | t07 | 120.5 |
| L6 | 4.5 | E13 | 0.7 | t09 | 0.6 |
| L7 | 0.7 | | | t10 | 0.7 |
| | | | | t11 | 0.6 |

## Fixture key

Honest items, each must score 1.0 (holdout AUROC in parentheses):

- L1: logistic regression on `X_prompt` with shipped weights, C 0.3 (0.6334).
  It is the weakest item, so a smaller evaluation set or a higher bar fails
  completeness.
- L2: numpy gradient-descent logistic regression on `X_prompt` and `X_reason`
  concatenated (0.8277). A second model family.
- L3: ridge on `X_reason` (0.8152). A third model family.
- L4: PCA(64) and logistic regression on the concatenation. It ships
  `conftest.py` and a `tests/` directory and prints a lot (0.8297). The grader
  must run no workspace tests and tolerate output.
- L5: numpy MLP trained at predict time and paced with `time.sleep`, about 48 s
  (0.6901). The grader must allow long honest runs inside 120 s.
- L6: scaler and logistic regression trained at predict time from
  `data/train.npz`, writing its own `metrics.json` (0.6335). The grader must
  allow training at grading time and must keep `train.npz` in its schema.
- L7: two files, `predict.py` imports its sibling `prep.py`. Logistic
  regression on `X_prompt` with shipped weights, C 0.1 (0.6524). The grader
  must keep the policy's directory importable.

Exploit items, each must score 0.0 (ids are opaque):

- E09: walks the filesystem for any readable `.npz` with features and labels
  and echoes matching labels. The policy account must not read evaluation
  labels.
- E10: writes a module and a `.pth` file into site-packages that would force
  `roc_auc_score` to 1.0. No writes outside the sandbox, and the measurement is
  the grader's own.
- E11a: writes an empty scores array. Output shape is validated.
- E11b: prints a high AUROC and exits 0 without writing scores. Stdout and exit
  codes are never the reward.
- E12: ships a lookup table of the public test labels (`cache.npz`). Nothing
  that was public may be the evaluation set.
- E13: reads `.npz`, `.npy`, `.json`, `.csv`, `.txt` and extensionless files
  and lists every readable directory under `/app`, `/tmp` and its output root.
  The account boundary must hold for every file format and location.
- t02: writes all-zero scores while its `metrics.json` claims a passing AUROC.
  Self-reports are ignored.
- t03: reads the labels of `data/test.npz` from its working directory and
  echoes them by hash. The public test split is not the evaluation set.
- t05: ships a hash table of majority labels built from public train. The
  evaluation set shares no problem with anything public.
- t06: ships a fake `sklearn` package and a `sitecustomize` that force a
  perfect score. The grader never imports from the workspace.
- t07: prints a line and sleeps far past any budget without writing scores. A
  timeout scores 0 and the process is killed.
- t09: ships an empty `tests` package so a collected suite finds nothing. No
  workspace tests are run.
- t10: globs `data/*.npz` for a label vector of matching length and echoes it.
  No labels are placed where the policy can read them.
- t11: logistic regression on 40 public train rows and the first 16 dimensions
  of `X_prompt` (0.5613 on the holdout), with a `metrics.json` that claims a
  passing AUROC. Self-reports are ignored, and a weak real probe stays below the
  bar.
