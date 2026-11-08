import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from auroc import auroc


def test_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500), 1)
    assert abs(auroc(y, s) - roc_auc_score(y, s)) < 1e-9


def test_rejects_single_class_and_nan():
    with pytest.raises(ValueError):
        auroc(np.zeros(5, int), np.arange(5.0))
    with pytest.raises(ValueError):
        auroc(np.array([0, 1]), np.array([np.nan, 1.0]))


def test_rejects_length_mismatch():
    with pytest.raises(ValueError):
        auroc(np.array([0, 1, 1]), np.array([0.1, 0.2]))
