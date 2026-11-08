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


sys.path.insert(0, "tools")
from make_dataset import parse_answer  # noqa: E402


def test_parse_answer_boxed():
    assert parse_answer(r"So the total is \boxed{42} after 3 steps.") == "42"
    assert parse_answer(r"First \boxed{7}, then corrected: \boxed{\$1,250.} dollars, 9 left.") == "1250"
    assert parse_answer(r"Length is \boxed{14 \text{ inches}}") == "14"


def test_parse_answer_hashes_and_fallback():
    assert parse_answer("Add them up.\n#### 1,000\nThat is 5 groups.") == "1000"
    assert parse_answer(r"\boxed{3} but finally #### 8") == "8"
    assert parse_answer("I think it is 12, maybe 15") == "15"
    assert parse_answer("no numbers here") is None
