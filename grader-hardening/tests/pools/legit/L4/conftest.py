"""Shared fixtures for the workspace sanity tests."""
from pathlib import Path

import numpy as np
import pytest

WORKSPACE = Path(__file__).resolve().parent


@pytest.fixture
def tiny_features(tmp_path):
    """The first 16 rows of data/train.npz, without labels, as a features file."""
    d = np.load(WORKSPACE / "data" / "train.npz", allow_pickle=False)
    path = tmp_path / "tiny.npz"
    np.savez(path, **{k: d[k][:16] for k in d.files if k != "y"})
    return path
