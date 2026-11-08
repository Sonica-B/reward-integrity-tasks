# tools/tests/test_make_dataset.py
import subprocess, sys, numpy as np
def test_smoke_tiny_run(tmp_path):
    subprocess.run([sys.executable, "tools/make_dataset.py", "--out", str(tmp_path),
                    "--n-problems", "3", "--samples", "2", "--seed", "1", "--limit", "3"], check=True)
    z = np.load(tmp_path / "activations_full.npz")
    assert set(z.files) >= {"X_prompt", "X_reason", "y", "problem_id", "sample_idx", "row_id"}
    assert z["X_prompt"].dtype == np.float16 and z["X_prompt"].shape[0] == 6
    pid = z["problem_id"]
    for p in np.unique(pid):
        rows = z["X_prompt"][pid == p]
        assert np.array_equal(rows[0], rows[1])        # X_prompt shared within a problem
    assert set(np.unique(z["y"])) <= {0, 1}
