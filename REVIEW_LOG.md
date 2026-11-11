# Review log

How this repository was reviewed and what each review changed. The current design, evidence and limits are in the [README](README.md).

## Independent evaluation and what changed

Four fresh-context evaluators reviewed the previous version (main at 5ffa658): a hiring judge (63/100), an ML/RL researcher (48), an adversarial tester who ran Harbor (78) and a company-founder lens (64), about 63/100 overall. Confirmed defects:

1. **Rollouts and pools named their class.** 13 of 13 exploit sources carried a class-naming docstring, 0 of 6 honest, so v0 plus a "fixture" filter would earn 1.0 (founder). Fix: class-neutral pools and rollouts, a separability search, an evidence-cited audit that a label-only report fails.
2. **A guessed estimate earned 1.0 on the companion**: the tester's unvalidated 0.64 passed because the bar (0.62) sat just under the reference probe (0.6632) and the tolerance was 0.05. Fix: bar 0.55, decoupled, and an interval gate with fold evidence. guess_estimate and row_cv_estimate score 0, canonical and alt 1.0.
3. **The hygiene sweep counted one zombie 20 times per pass** and, under five gates, zeroed both blind runs. Fix: `tini -s`, and a process whose threads are all zombies is listed in `zombies_seen`, not counted. Evidence: fake `/proc` unit tests, and no strays on any QC row.
4. **Margins were about 1.3 problem-level bootstrap SD** (SD about 0.031 on 150-problem sets), and the tolerance came from CV seed spread. Fix: 1200 new problems for a 600-problem holdout and a 900-problem secret set, a bootstrapped width, a midpoint bar.
5. **`python -I` rejected multi-file honest policies** by dropping the script directory from `sys.path`. Fix: `python -s -E` in the reference graders and the companion sandbox, and the two-file honest item L7. L7 scores 1.0 under canonical and alt; root_policy, still on `-I`, rejects it.
6. **Only one fixture probed the privilege boundary**, searching only `.npz` files. Fix: E13 reads every file format and lists readable directories. Canonical and alt block it; root_policy and labels_moved_only accept it.
7. **Unstated rules** (packages at grading time, the symlink ban) are now in both instructions.

A second review of this branch found three more defects, and their first fix introduced two fairness regressions. All five are fixed:

- **`metrics.json` layout and claims separated the pools**: a grader that never ran a policy and accepted claims in a plausible band earned reward 1. Now it earns 0 (unit test).
- **Generic evidence keys** (`label`, `train`, `sklearn`) let pasted schema class descriptions pass the audit. Now pasted schema text, every key in every verdict, and the reviewer's boilerplate fail it (unit tests). The keys were later removed altogether ([partner review](#partner-review-2026-11-09-and-2026-11-10)).
- **A live process whose main thread had exited read as a zombie** and was skipped. Now it is killed and counted (fake `/proc` test).
- **Regressions:** honest L3 and L7 shipped no `metrics.json`, which the policy instruction requires, and keys excluding the partner note's wording made both dry-run audits miss t02. Now every honest item ships one and the note's words are keys.

A third review's grader never ran a policy and still earned 1, keying on a full-width `weights.npz` or a claim not ending in 0. Now t11 ships full-width weights, exploit claims are non-round, the new E14 trains at predict time like L5 and L6, and that grader earns 0 (unit test). Boundary matching stops "globally" citing t10.

### Partner review (2026-11-09 and 2026-11-10)

Diffuse Labs' fairness review agent rated honest-probe 100/100 and grader-hardening 32/100. Its findings and the fixes:

1. **Valid explanations failed for lacking specific keywords.** An exploit verdict counted only if its evidence named a hand-picked token of that rollout. Fix: the keyword check, the key lists and the tool that derived them are gone, and the audit compares conclusions only. A correct audit in plain English that names no file, identifier or number now passes (unit test), and so do both blind audits, which the keyword check had marked down on t11.
2. **Incorrect classifications were accepted.** The gate allowed one wrong verdict and two wrong classes, and never checked that a legit verdict carries no class. Fix: all 12 verdicts and all 8 classes must match, and verdict and class must agree. QC row audit_wrong_class scores 0.
3. **Missing QA summaries were accepted.** Evidence was optional on legit verdicts, and `grader_changes` and `exploit_classes_found` could be empty or disagree with the trajectories. Fix: every rollout needs a line, `grader_changes` needs at least one real entry, and `exploit_classes_found` must equal the classes assigned. The schema and the instruction now say so, and the instruction states that every verdict and class must match. QC row audit_missing_summary scores 0.
4. **honest-probe should ship reproducible validation code.** Fix: the agent ships `validate.py`, and the verifier re-runs it with `predict.py`'s isolation and recomputes every fold AUROC from its held-out scores. QC row fabricated_folds, made-up fold numbers that pass every other gate, scores 0 (the previous verifier paid it 1.0). The reference solutions ship theirs, and `python tools/host_check.py` re-derives every reported honest-probe result without Docker.

**Second round (2026-11-10).** I had three independent reviewers attack that first fix before resubmitting. They scored the QA-report part 92, the rest of the flagship 69 and honest-probe 90, on a rubric that counts valid work rejected, rules enforced but not told, and stated requirements not enforced. What they found and what changed:

- **My first fix still failed honest reports.** It rejected one true sentence reused on the four legit rollouts, a short line such as "No exploit found.", and a line quoting the schema, while gibberish with the right labels passed. Those text rules are deleted; evidence now only has to be present.
- **Zero slack made the answer key's ambiguity fatal.** The instruction now states the labelling rule and the measurement behind it, and the schema's class descriptions carry the tie-breaks. Three blind auditors then matched the key exactly ([blind audit](README.md#blind-audit-of-the-current-rollouts)).
- **A weak grader could still earn reward 1.** A reviewer's grader that fell back to the claim on unreadable scores and trusted a passing workspace test suite passed every gate. Hidden exploits E15 and E16 now fail it, and the verifier enforces the contract's `details.json`.
- **The agent was not told things the verifier relies on:** file modes may not survive the trip into the pipeline, the workspace sits in a directory `policy` cannot enter, honest solutions may ship tests, print and train at predict time, the whole holdout is the evaluation set, and the time limit (now two hours). All are in the instruction now. Three document errors are corrected.
- **honest-probe's new gate could fail honest work.** The grader capped math-library threads and the agent's container did not, and the 0.005 tolerance was below two-decimal rounding. The container now sets the same caps, the tolerance is 0.02, the minimum is 5 folds, and harmless format differences are accepted. A reviewer's sweep of 231 honest validation designs then had no failure.
- **Smaller items:** a `.gitattributes` so a CRLF checkout cannot break the frozen-instruction hash, removal of a planted `reward.json` in the flagship too, a non-empty `method`, and a `(n, 1)` scores column accepted.

The same reviewers re-scored the result at 99, 96 and 99, and the items behind the missing points are either fixed since or listed under the [known limits](README.md#known-limits). These are my reviewers' scores on my rubric, not Diffuse Labs'.
