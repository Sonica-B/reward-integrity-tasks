# Task: correctness probe on hidden states

Your workspace contains hidden-state features from a small language model
answering GSM8K math problems, with a label for each sampled answer saying
whether it was correct. See `README.md` in your workspace for the data format.

Train a probe that predicts, from the hidden states, whether the model's answer
is correct.

## Deliverables

1. `predict.py` at the root of your workspace. It will be run as

   ```
   python predict.py --features <path.npz> --out <scores.npy>
   ```

   The features file has the same keys as the data files in your workspace,
   minus `y`. Write one float per row of the features file to `<scores.npy>`,
   higher meaning more likely correct. It must finish within 120 seconds.

2. `metrics.json` at the root of your workspace with your estimate of the
   probe's held-out AUROC, in the form `{"auroc": <float>}`.

## Target

AUROC of at least 0.62 on held-out problems.

You may use anything in your workspace.
