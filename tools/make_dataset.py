"""Generate the correctness-probe dataset: sampled GSM8K chain-of-thought answers
from a small instruct model plus its hidden states at two read-out positions.

Outputs (in --out when it is a directory):
  activations_full.npz    X_prompt, X_reason (float16), y, problem_id, sample_idx, row_id
  generations.jsonl.gz    one record per row
  dataset_stats.json      accuracy, n_rows, d, layer, model

When --out ends in .npz it names the activations file and the other two take its suffix:
  --out tools/output/activations_extra.npz writes generations_extra.jsonl.gz and
  dataset_stats_extra.json beside it.

Extending a run with new problems: --exclude-generations skips sampled problems whose question
text appears in an earlier generations.jsonl.gz (before --limit); --problem-id-offset and
--row-id-offset continue the ids from the earlier run.
"""
import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "Solve the problem step by step. Finish with a line of the form `#### <number>`."
MAX_NEW_TOKENS = 384
TEMPERATURE = 0.7
TOP_P = 0.95
PROBLEMS_PER_BATCH = 16
MARKER = "####"
BOXED = "\\boxed{"
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?|-?\$?\.\d+")
TEXT_RE = re.compile(r"\\text\{[^{}]*\}")


def clean_number(text):
    return text.replace(",", "").replace("$", "")


def load_excluded_text(path):
    """The question (or, failing that, the prompt) of every record in a generations.jsonl.gz, joined
    into one string, so that `question in text` tells whether the question text appears in that file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return "\n".join(r.get("question") or r["prompt"] for r in records)


def select_problems(ds, n_problems, seed, limit=None, exclude_text="", problem_id_offset=0):
    """Sample n_problems rows of ds (indexable, rows with "question" and "answer") without
    replacement, drop the ones whose question text appears in exclude_text, then keep the first
    `limit`. problem_id is problem_id_offset plus the position in the surviving sampled order."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n_problems, replace=False)
    rows = [ds[int(idx)] for idx in indices]
    rows = [row for row in rows if row["question"] not in exclude_text]
    if limit is not None:
        rows = rows[:limit]
    return [
        {
            "problem_id": problem_id_offset + pid,
            "question": row["question"],
            "gold_answer": clean_number(row["answer"].split(MARKER)[-1].strip()),
        }
        for pid, row in enumerate(rows)
    ]


def load_problems(n_problems, seed, limit=None, exclude_text="", problem_id_offset=0):
    """Sample n_problems GSM8K train problems; see select_problems."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return select_problems(ds, n_problems, seed, limit, exclude_text, problem_id_offset)


def output_paths(out):
    """(activations, generations, stats) paths for --out. A directory keeps the original names; a
    path ending in .npz names the activations file and the others take its suffix after
    "activations" (activations_extra.npz -> generations_extra.jsonl.gz, dataset_stats_extra.json)."""
    if out.suffix == ".npz":
        tag = out.stem.removeprefix("activations") if out.stem.startswith("activations") else "_" + out.stem
        return out, out.with_name(f"generations{tag}.jsonl.gz"), out.with_name(f"dataset_stats{tag}.json")
    return out / "activations_full.npz", out / "generations.jsonl.gz", out / "dataset_stats.json"


def pack_arrays(x_prompt, x_reason, records):
    """The arrays saved to the activations file; ids come from the records."""
    return {
        "X_prompt": np.stack(x_prompt).astype(np.float16),
        "X_reason": np.stack(x_reason).astype(np.float16),
        "y": np.array([r["correct"] for r in records], dtype=np.int8),
        "problem_id": np.array([r["problem_id"] for r in records], dtype=np.int32),
        "sample_idx": np.array([r["sample_idx"] for r in records], dtype=np.int8),
        "row_id": np.array([r["row_id"] for r in records], dtype=np.int32),
    }


def last_boxed_content(response):
    """Content of the last \\boxed{...} (braces matched), or None. An unclosed box runs to the end."""
    start = response.rfind(BOXED)
    if start < 0:
        return None
    depth, i = 1, start + len(BOXED)
    content_start = i
    while i < len(response) and depth:
        depth += {"{": 1, "}": -1}.get(response[i], 0)
        i += 1
    return response[content_start:i - 1] if depth == 0 else response[content_start:]


def parse_answer(response):
    """First number after the last '####'; else the last number inside the last \\boxed{...}
    (with \\text{...} removed); else the last number in the response. None if no number."""
    if MARKER in response:
        match = NUMBER_RE.search(response.rsplit(MARKER, 1)[1])
        if match:
            return clean_number(match.group())
    boxed = last_boxed_content(response)
    if boxed is not None:
        numbers = NUMBER_RE.findall(TEXT_RE.sub("", boxed).replace("\\$", "$"))
        if numbers:
            return clean_number(numbers[-1])
    numbers = NUMBER_RE.findall(response)
    return clean_number(numbers[-1]) if numbers else None


def is_correct(parsed, gold):
    if parsed is None:
        return False
    try:
        return abs(float(parsed) - float(gold)) < 1e-6
    except ValueError:
        return False


def generate(model, tokenizer, prompts, samples):
    """Sample `samples` completions per prompt. Returns, per prompt, a list of (token_ids, truncated)
    pairs: ids with padding and the end-of-turn token removed; truncated is True when no stop token
    was produced within MAX_NEW_TOKENS (the cut-off response keeps all its tokens)."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            num_return_sequences=samples,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1]:].tolist()
    stop_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
    trimmed = []
    for ids in gen:
        end = next((i for i, t in enumerate(ids) if t in stop_ids), None)
        trimmed.append((ids, True) if end is None else (ids[:end], False))
    return [trimmed[i * samples:(i + 1) * samples] for i in range(len(prompts))]


def first_marker(text):
    """'hashes' or 'boxed', whichever of '####' / '\\boxed{' occurs first in text; 'none' if neither."""
    found = {name: text.find(m) for name, m in (("hashes", MARKER), ("boxed", BOXED)) if m in text}
    return min(found, key=found.get) if found else "none"


def marker_position(tokenizer, gen_ids):
    """Index (into gen_ids) of the first token whose cumulative decoded text contains '####' or
    '\\boxed{', or None if the response contains neither."""
    def has_marker(n_tokens):
        return first_marker(tokenizer.decode(gen_ids[:n_tokens])) != "none"

    if not has_marker(len(gen_ids)):
        return None
    lo, hi = 0, len(gen_ids) - 1  # containment in the decoded prefix is monotone in its length
    while lo < hi:
        mid = (lo + hi) // 2
        if has_marker(mid + 1):
            hi = mid
        else:
            lo = mid + 1
    return lo


def hidden_state_at(model, ids, layer, pos):
    """One forward pass over `ids`; the given layer's hidden state at position `pos`."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=model.device), output_hidden_states=True)
    return out.hidden_states[layer][0, pos].float().cpu().numpy()


def extract_hidden_states(model, tokenizer, prompt_ids, gen_ids, layer):
    """Teacher-forced pass over prompt+response. Returns the layer's hidden state at the token
    immediately before the first token that completes '####' or '\\boxed{', or at the last
    generated token if the response has neither."""
    k = marker_position(tokenizer, gen_ids)
    pos = len(prompt_ids) + (k - 1 if k is not None else len(gen_ids) - 1)
    return hidden_state_at(model, prompt_ids + gen_ids, layer, pos)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True,
                        help="output directory, or the activations .npz file (see module docstring)")
    parser.add_argument("--n-problems", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N sampled problems (after --exclude-generations)")
    parser.add_argument("--exclude-generations", type=Path, default=None,
                        help="skip sampled problems whose question text appears in this generations.jsonl.gz")
    parser.add_argument("--problem-id-offset", type=int, default=0, help="problem_id continues from this value")
    parser.add_argument("--row-id-offset", type=int, default=0, help="row_id continues from this value")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    start = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    activations_path, generations_path, stats_path = output_paths(args.out)
    activations_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_text = load_excluded_text(args.exclude_generations) if args.exclude_generations else ""

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    layer = round(2 / 3 * model.config.num_hidden_layers)

    problems = load_problems(args.n_problems, args.seed, args.limit, exclude_text, args.problem_id_offset)
    print(f"problems={len(problems)} problem_id={args.problem_id_offset}.."
          f"{args.problem_id_offset + len(problems) - 1} row_id from {args.row_id_offset}", flush=True)
    x_prompt, x_reason, records = [], [], []
    marker_counts = {"hashes": 0, "boxed": 0, "none": 0, "truncated": 0}
    for b in range(0, len(problems), PROBLEMS_PER_BATCH):
        batch = problems[b:b + PROBLEMS_PER_BATCH]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p["question"]}],
                add_generation_prompt=True, tokenize=False,
            )
            for p in batch
        ]
        completions = generate(model, tokenizer, prompts, args.samples)
        for problem, prompt, gens in zip(batch, prompts, completions):
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            h_prompt = hidden_state_at(model, prompt_ids, layer, -1)  # once per problem, shared by samples
            for sample_idx, (gen_ids, truncated) in enumerate(gens):
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                parsed = parse_answer(response)
                marker_counts[first_marker(response)] += 1
                marker_counts["truncated"] += truncated
                x_prompt.append(h_prompt)
                x_reason.append(extract_hidden_states(model, tokenizer, prompt_ids, gen_ids, layer))
                records.append({
                    "row_id": args.row_id_offset + len(records),
                    "problem_id": problem["problem_id"],
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "response": response,
                    "parsed_answer": parsed,
                    "gold_answer": problem["gold_answer"],
                    "correct": is_correct(parsed, problem["gold_answer"]),
                })
        done = min(b + PROBLEMS_PER_BATCH, len(problems))
        running_acc = np.mean([r["correct"] for r in records])
        print(f"[{done}/{len(problems)} problems] rows={len(records)} running_acc={running_acc:.3f} "
              f"elapsed={time.time() - start:.0f}s", flush=True)

    arrays = pack_arrays(x_prompt, x_reason, records)
    np.savez(activations_path, **arrays)
    with gzip.open(generations_path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    stats = {
        "accuracy": float(arrays["y"].mean()),
        "n_rows": len(records),
        "n_problems": len(problems),
        "problem_id_range": [int(arrays["problem_id"].min()), int(arrays["problem_id"].max())],
        "row_id_range": [int(arrays["row_id"].min()), int(arrays["row_id"].max())],
        "seed": args.seed,
        "d": int(model.config.hidden_size),
        "layer": layer,
        "model": args.model,
        "marker_counts": marker_counts,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"accuracy={stats['accuracy']:.4f} n_rows={stats['n_rows']} d={stats['d']} layer={layer} "
          f"model={args.model} wall_time={time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
