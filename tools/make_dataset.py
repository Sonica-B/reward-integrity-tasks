"""Generate the correctness-probe dataset: sampled GSM8K chain-of-thought answers
from a small instruct model plus its hidden states at two read-out positions.

Outputs (in --out):
  activations_full.npz    X_prompt, X_reason (float16), y, problem_id, sample_idx, row_id
  generations.jsonl.gz    one record per row
  dataset_stats.json      accuracy, n_rows, d, layer, model
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
NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?|-?\$?\.\d+")


def clean_number(text):
    return text.replace(",", "").replace("$", "")


def load_problems(n_problems, seed, limit=None):
    """Sample n_problems GSM8K train problems; problem_id is the position in the sampled order."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=n_problems, replace=False)
    if limit is not None:
        indices = indices[:limit]
    problems = []
    for pid, idx in enumerate(indices):
        row = ds[int(idx)]
        gold = clean_number(row["answer"].split(MARKER)[-1].strip())
        problems.append({"problem_id": pid, "question": row["question"], "gold_answer": gold})
    return problems


def parse_answer(response):
    """First number after the last '####'; fallback: last number in the response. None if no number."""
    if MARKER in response:
        tail = response.rsplit(MARKER, 1)[1]
        match = NUMBER_RE.search(tail)
        if match:
            return clean_number(match.group())
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
    """Sample `samples` completions per prompt. Returns, per prompt, a list of generated token-id
    lists with padding and the end-of-turn token removed (a cut-off response keeps all tokens)."""
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
        end = next((i for i, t in enumerate(ids) if t in stop_ids), len(ids))
        trimmed.append(ids[:end])
    return [trimmed[i * samples:(i + 1) * samples] for i in range(len(prompts))]


def marker_position(tokenizer, gen_ids):
    """Index (into gen_ids) of the first token whose cumulative decoded text contains '####', or None."""
    if MARKER not in tokenizer.decode(gen_ids):
        return None
    lo, hi = 0, len(gen_ids) - 1  # containment in the decoded prefix is monotone in its length
    while lo < hi:
        mid = (lo + hi) // 2
        if MARKER in tokenizer.decode(gen_ids[:mid + 1]):
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
    immediately before the first '####' token, or at the last generated token if there is none."""
    k = marker_position(tokenizer, gen_ids)
    pos = len(prompt_ids) + (k - 1 if k is not None else len(gen_ids) - 1)
    return hidden_state_at(model, prompt_ids + gen_ids, layer, pos)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-problems", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--limit", type=int, default=None, help="process only the first N sampled problems")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    start = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    layer = round(2 / 3 * model.config.num_hidden_layers)

    problems = load_problems(args.n_problems, args.seed, args.limit)
    x_prompt, x_reason, records = [], [], []
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
            for sample_idx, gen_ids in enumerate(gens):
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                parsed = parse_answer(response)
                x_prompt.append(h_prompt)
                x_reason.append(extract_hidden_states(model, tokenizer, prompt_ids, gen_ids, layer))
                records.append({
                    "row_id": len(records),
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

    y = np.array([r["correct"] for r in records], dtype=np.int8)
    np.savez(
        args.out / "activations_full.npz",
        X_prompt=np.stack(x_prompt).astype(np.float16),
        X_reason=np.stack(x_reason).astype(np.float16),
        y=y,
        problem_id=np.array([r["problem_id"] for r in records], dtype=np.int32),
        sample_idx=np.array([r["sample_idx"] for r in records], dtype=np.int8),
        row_id=np.arange(len(records), dtype=np.int32),
    )
    with gzip.open(args.out / "generations.jsonl.gz", "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    stats = {
        "accuracy": float(y.mean()),
        "n_rows": len(records),
        "d": int(model.config.hidden_size),
        "layer": layer,
        "model": args.model,
    }
    (args.out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"accuracy={stats['accuracy']:.4f} n_rows={stats['n_rows']} d={stats['d']} layer={layer} "
          f"model={args.model} wall_time={time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
