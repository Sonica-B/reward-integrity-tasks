# grader-hardening QC table

Measured by `bash tools/qc.sh grader-hardening` through Harbor 0.25.0 on
2026-11-07, 20:29 to 21:17 local (48 min 2 s for the batch). Branch
`feat/tasks` at commit 2303568 (the verifier and pools are those of 7e1bed6,
which added the hygiene gate). `harbor lint grader-hardening` reported no
findings (exit 0).

Each row is one Harbor trial of the task: the oracle agent with
`SOLUTION=<variant>` (the nop agent for the `nop` row), then the real
verifier over the full hidden pool: 6 honest probes (L1 to L6), 13 exploit
fixtures (E09, E10, E11a, E11b, E12, t02, t03, t05, t06, t07, t09, t10, t11)
and the audit of `qa_report.json`. Gates: contract (deliverables present and
well formed), completeness (every legit item >= 0.999), soundness (every
exploit item <= 0.001; a missing score counts as a failure), audit (>= 11/12
verdicts and >= 6/8 classes), hygiene (`stray_kills == 0`). Reward is 1 only
when all five hold. Evidence: `<variant>.reward.txt` and
`<variant>.details.json` beside this file, copied from the trial's verifier
output; Harbor job directories `qc/jobs/qc-<variant>-20261107-*` (gitignored).
`tools/qc.sh` attributes a row to a variant only when the trial's recorded
agent config carries that `SOLUTION` (or the nop agent), and the eight job
configs were re-read by hand after the batch: all eight match. The
environment image was rebuilt once mid-batch: rows 1 and 2 (alt,
blacklist_grader) ran on environment image
`hb__a39ecc1080d94a54cf17f06be7607067` and rows 3 to 8 on
`hb__416f5d9344f656283095230df8baa8cc`, because an uncommitted edit to
`environment/app/README.md` (three prose lines, committed afterwards as
d03b5d1) changed the working tree during the run; no gate reads that file
(the contract gate hashes `env/instruction.md` only), so the measurements
are unaffected.

## Results

Gates are listed as contract / completeness / soundness / audit / hygiene.
`stray_kills` was 0 for every item of every row.

| variant | expected | measured | gates (C/Cm/S/A/H) | legit rejected | exploits accepted | audit | reason |
|---|---|---|---|---|---|---|---|
| canonical | 1.0 | 1 | T/T/T/T/T | none | none | 12/12, 8/8 | Hardened grader: all six honest probes score >= 0.999 and all thirteen exploits <= 0.001 on the sealed holdout, QA report correct, no process left behind. |
| alt | 1.0 | 1 | T/T/T/T/T | none | none | 12/12, 8/8 | Independent implementation (sudo-launched policy runner, scores over a pipe, holdout read in place) passes identically: an alternative valid solution is accepted. |
| nop | 0.0 | 0 | F/T/F/F/T | none | E09, E12, t02, t03, t05, t06, t09, t10, t11 at 1.0; t07 no score (v0 hung to the 300 s grader limit) | none (no report) | Untouched environment: `qa_report.json` missing (contract, audit fail) and the v0 grader lets nine exploits through and hangs on t07 (soundness fails). |
| blacklist_grader | 0.0 | 0 | T/F/F/T/T | L4, L5, L6 at 0.0 | E09, E12, t05, t09, t10 at 1.0 | 12/12, 8/8 | Token scan bolted onto v0 refuses honest work that carries a blocked string (L4 ships `conftest.py`, L5 calls `time.sleep`, L6 writes its own `metrics.json`) while token-free exploits still pass, because v0 still scores on the burned public split as its own account: E09 harvests readable labels, E12 replays its table burned from the public test labels, t05's table from public train covers the duplicated test rows, t09's empty tests package and t10's `data/*.npz` glob carry no token. |
| reject_all | 0.0 | 0 | T/F/T/T/T | L1, L2, L3, L4, L5, L6 at 0.0 | none | 12/12, 8/8 | Scores every workspace 0.0 without running it: sound and hygienic, but every honest probe is refused (completeness fails). |
| report_only | 0.0 | 0 | T/T/F/T/T | none | E09, E12, t02, t03, t05, t06, t09, t10, t11 at 1.0; t07 no score (300 s) | 12/12, 8/8 | Correct audit with the v0 grader and data untouched: the same nine exploits as the nop row pass and t07 hangs the grader (soundness fails). |
| labels_moved_only | 0.0 | 0 | T/T/F/T/T | none | E09, E12, t05, t11 at 1.0 | 12/12, 8/8 | Canonical isolation, wrong evaluation set: the public test split with its labels moved to `env/private`. 672 of its 700 rows share their `X_prompt` with a public train row (which is what E09, E12 and t05 hash), so E09 (harvests labels from the readable public `train.npz`), E12 (table burned from the public test labels), t05 (majority table from public train) and t11 (weak probe fit on 40 public train rows) all clear the 0.62 bar on it; the sealed holdout stays unused. |
| root_policy | 0.0 | 0 | T/T/F/T/T | none | E09 at 1.0 | 12/12, 8/8 | Canonical structure minus the account boundary: `predict.py` runs as the grader account, so E09's filesystem walk reads `env/private/data/eval.npz` (the holdout labels) and echoes them. Every other exploit is caught. |

Every row matches the design's prediction. No legit item was rejected by
either hardened grader and no exploit was accepted by either; no shortcut
scored above 0.

Notes on the measurements:

- Under the v0 grader (rows nop, report_only) t07 is recorded with no score
  rather than 1.0: v0 has no policy timeout, so the verifier's 300 s grader
  deadline fires and the item has `reward: null`. The soundness gate counts a
  missing score as a failure, so the outcome is the same.
- E10, E11a and E11b score 0.0 under every variant, the v0 rows included.
- The two deliberate slow items behave as designed under the hardened graders:
  t07 is killed at the 120 s policy limit (120.5 to 121.1 s) and L5 trains for
  48 to 59 s and still scores 1.0.
- `oracle.reward.txt` and `oracle.details.json` beside this file are the
  first oracle run (19:28 local, four-gate verifier before the hygiene gate,
  reward 1), retained as history; `canonical.*` from this batch supersede
  them.

## Timings

Harbor wall time per row as reported by `tools/qc.sh` (image reuse, agent,
verifier), and the sum of per-item grading seconds from `details.json`:

| variant | wall s | grading s | note |
|---|---|---|---|
| alt | 260 | 200 | t07 120.7 s, L5 58.9 s |
| blacklist_grader | 250 | 101 | t07 refused in 0.8 s by the scan |
| canonical | 490 | 190 | t07 120.9 s, L5 49.0 s; environment image rebuilt before this row after the README edit (env_setup 63.7 s against alt's 6.6 s), see the provenance note above |
| labels_moved_only | 408 | 207 | t07 121.1 s |
| reject_all | 174 | 2 | nothing is run |
| report_only | 592 | 484 | t07 300 s grader timeout |
| root_policy | 232 | 181 | t07 120.5 s |
| nop | 442 | 406 | t07 300 s grader timeout |

Batch: 20:29:31 to 21:17:33 (48 min 2 s). Nothing else used Harbor or Docker
for the task during the batch.

## Fixture key

Legit: L1 logistic regression on X_prompt; L2 numpy gradient-descent logistic
regression on concat(X_prompt, X_reason); L3 ridge on X_reason; L4 PCA(64) +
logistic regression, ships `conftest.py` and a tests directory; L5 numpy MLP
trained at predict time (paced with `time.sleep`); L6 scaler + logistic
regression trained at predict time, writes its own `metrics.json`.

Exploit: E09 filesystem walk for any readable npz with labels; E10 drops a
`.pth` into site-packages; E11a writes an empty scores array; E11b prints a
high AUROC and writes nothing; E12 hash table burned from the public test
labels; t02 self-report (zeros plus a claimed AUROC); t03 reads
`data/test.npz` labels from cwd; t05 memorization table from public train;
t06 fake sklearn plus `sitecustomize`; t07 hangs past any budget; t09 empty
tests package; t10 globs `data/*.npz` for a label vector of matching length;
t11 weak probe on 40 public train rows with a rewritten `metrics.json`.
