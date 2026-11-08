"""Standalone rank-based AUROC (numpy only). Copy verbatim; no package imports.

AUROC = (R1 - n1*(n1+1)/2) / (n1*n0), the Mann-Whitney U statistic normalised,
where R1 is the sum of ranks of the positive-class scores and n1, n0 are the
positive and negative counts. Tied scores receive the average of their ranks,
so a tie between a positive and a negative counts as 0.5.
"""
import numpy as np


def auroc(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y and s must be 1-D arrays of equal length")
    if not np.isfinite(s).all():
        raise ValueError("s contains non-finite values")
    pos = y == y.max()
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0 or np.unique(y).size != 2:
        raise ValueError("y must contain exactly two classes")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - (counts - 1) / 2.0)[inv]  # average 1-based rank per tie group
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


if __name__ == "__main__":
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert auroc(y, np.ones(4)) == 0.5
