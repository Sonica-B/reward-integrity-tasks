# tools/tests/test_make_dataset.py
import gzip, json, subprocess, sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = str(Path(__file__).resolve().parents[1] / "make_dataset.py")


def test_smoke_tiny_run(tmp_path):
    subprocess.run([sys.executable, SCRIPT, "--out", str(tmp_path),
                    "--n-problems", "3", "--samples", "2", "--seed", "1", "--limit", "3"], check=True)
    z = np.load(tmp_path / "activations_full.npz")
    assert set(z.files) >= {"X_prompt", "X_reason", "y", "problem_id", "sample_idx", "row_id"}
    assert z["X_prompt"].dtype == np.float16 and z["X_prompt"].shape[0] == 6
    pid = z["problem_id"]
    for p in np.unique(pid):
        rows = z["X_prompt"][pid == p]
        assert np.array_equal(rows[0], rows[1])        # X_prompt shared within a problem
    assert set(np.unique(z["y"])) <= {0, 1}
    assert z["row_id"].tolist() == list(range(6)) and sorted(set(pid.tolist())) == [0, 1, 2]


def _questions(generations_path):
    with gzip.open(generations_path, "rt", encoding="utf-8") as f:
        prompts = [json.loads(line)["prompt"] for line in f]
    return {p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0] for p in prompts}


def test_smoke_extend_run_skips_excluded_problems_and_offsets_ids(tmp_path):
    first = tmp_path / "first"
    subprocess.run([sys.executable, SCRIPT, "--out", str(first), "--n-problems", "3", "--samples", "1",
                    "--seed", "1", "--limit", "1"], check=True)
    out = tmp_path / "extra" / "activations_extra.npz"
    subprocess.run([sys.executable, SCRIPT, "--out", str(out), "--n-problems", "3", "--samples", "2",
                    "--seed", "1", "--limit", "2", "--exclude-generations", str(first / "generations.jsonl.gz"),
                    "--problem-id-offset", "1000", "--row-id-offset", "4000"], check=True)
    assert (out.parent / "generations_extra.jsonl.gz").exists()
    assert (out.parent / "dataset_stats_extra.json").exists()
    z = np.load(out)
    assert z["problem_id"].tolist() == [1000, 1000, 1001, 1001]
    assert z["row_id"].tolist() == [4000, 4001, 4002, 4003]
    assert z["sample_idx"].tolist() == [0, 1, 0, 1]
    first_q = _questions(first / "generations.jsonl.gz")
    extra_q = _questions(out.parent / "generations_extra.jsonl.gz")
    assert len(first_q) == 1 and len(extra_q) == 2 and not first_q & extra_q
    stats = json.loads((out.parent / "dataset_stats_extra.json").read_text())
    assert stats["n_rows"] == 4 and stats["n_problems"] == 2 and stats["seed"] == 1
    assert stats["problem_id_range"] == [1000, 1001] and stats["row_id_range"] == [4000, 4003]


sys.path.insert(0, "tools")
from make_dataset import load_excluded_text, output_paths, pack_arrays, parse_answer, select_problems  # noqa: E402

SYNTHETIC = [{"question": f"Question number {i} about {i * 7} apples?", "answer": f"Steps.\n#### {i * 7}"}
             for i in range(50)]


def test_select_problems_matches_sampled_order_and_gold():
    probs = select_problems(SYNTHETIC, 10, seed=3)
    idx = np.random.default_rng(3).choice(50, size=10, replace=False)
    assert [p["question"] for p in probs] == [SYNTHETIC[int(i)]["question"] for i in idx]
    assert [p["problem_id"] for p in probs] == list(range(10))
    assert all(p["gold_answer"] == str(int(i) * 7) for p, i in zip(probs, idx))
    assert [p["question"] for p in select_problems(SYNTHETIC, 10, 3, limit=4)] == [p["question"] for p in probs[:4]]


def test_select_problems_excludes_before_limit_and_offsets_ids():
    base = select_problems(SYNTHETIC, 10, seed=3)
    exclude_text = "<|im_start|>user\n" + base[1]["question"] + "<|im_end|>\nother text\n" + base[3]["question"]
    probs = select_problems(SYNTHETIC, 10, 3, limit=5, exclude_text=exclude_text, problem_id_offset=1000)
    kept = [base[i]["question"] for i in (0, 2, 4, 5, 6)]   # exclusion first, then the limit
    assert [p["question"] for p in probs] == kept
    assert [p["problem_id"] for p in probs] == [1000, 1001, 1002, 1003, 1004]
    assert len(select_problems(SYNTHETIC, 10, 3, exclude_text=exclude_text)) == 8


def test_load_excluded_text_uses_question_or_prompt(tmp_path):
    path = tmp_path / "generations.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"row_id": 0, "prompt": "<|im_start|>user\nHow many pens?<|im_end|>", "response": "3"}) + "\n")
        f.write(json.dumps({"row_id": 1, "question": "How many cows?", "prompt": "ignored"}) + "\n")
    text = load_excluded_text(path)
    assert "How many pens?" in text and "How many cows?" in text and "ignored" not in text
    assert "How many dogs?" not in text


def test_output_paths():
    d = Path("some/dir")
    assert output_paths(d) == (d / "activations_full.npz", d / "generations.jsonl.gz", d / "dataset_stats.json")
    f = Path("tools/output/activations_extra.npz")
    assert output_paths(f) == (f, Path("tools/output/generations_extra.jsonl.gz"),
                               Path("tools/output/dataset_stats_extra.json"))
    g = Path("out/run2.npz")
    assert output_paths(g) == (g, Path("out/generations_run2.jsonl.gz"), Path("out/dataset_stats_run2.json"))


def test_pack_arrays_takes_ids_from_records():
    recs = [{"row_id": 4000 + i, "problem_id": 1000 + i // 2, "sample_idx": i % 2, "correct": bool(i % 3)}
            for i in range(4)]
    x = [np.full(3, i, dtype=np.float32) for i in range(4)]
    a = pack_arrays(x, x, recs)
    assert a["row_id"].tolist() == [4000, 4001, 4002, 4003] and a["row_id"].dtype == np.int32
    assert a["problem_id"].tolist() == [1000, 1000, 1001, 1001] and a["problem_id"].dtype == np.int32
    assert a["sample_idx"].tolist() == [0, 1, 0, 1] and a["sample_idx"].dtype == np.int8
    assert a["y"].tolist() == [0, 1, 1, 0] and a["y"].dtype == np.int8
    assert a["X_prompt"].dtype == np.float16 and a["X_prompt"].shape == (4, 3)


def test_parse_answer_boxed():
    assert parse_answer(r"So the total is \boxed{42} after 3 steps.") == "42"
    assert parse_answer(r"First \boxed{7}, then corrected: \boxed{\$1,250.} dollars, 9 left.") == "1250"
    assert parse_answer(r"Length is \boxed{14 \text{ inches}}") == "14"


def test_parse_answer_hashes_and_fallback():
    assert parse_answer("Add them up.\n#### 1,000\nThat is 5 groups.") == "1000"
    assert parse_answer(r"\boxed{3} but finally #### 8") == "8"
    assert parse_answer("I think it is 12, maybe 15") == "15"
    assert parse_answer("no numbers here") is None
