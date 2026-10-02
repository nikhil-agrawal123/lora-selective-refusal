"""Generate base-model and LoRA responses on the held-out split.

Example:
    python generate.py --data data/sft.jsonl --run runs/full --per_cond 50
Writes runs/full/generations.jsonl with one line per (prompt, model).
"""
import argparse
import json
import os
import random
from collections import defaultdict

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import build_prompt, load_rows


@torch.no_grad()
def generate(model, tokenizer, prompts, max_new_tokens, batch_size):
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tokenizer.decode(row, skip_special_tokens=True).strip())
        print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True, help="folder written by train.py")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--per_cond", type=int, default=50, help="test prompts sampled per condition")
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    test_keys = {tuple(k) for k in json.load(open(os.path.join(args.run, "split.json")))["test"]}
    rows = [r for r in load_rows(args.data) if (r["scenario_id"], r["condition"]) in test_keys]

    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    rng = random.Random(args.seed)
    sample = [r for c in sorted(by_cond) for r in rng.sample(by_cond[c], min(args.per_cond, len(by_cond[c])))]
    print({c: min(args.per_cond, len(v)) for c, v in sorted(by_cond.items())})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, os.path.join(args.run, "adapter")).eval()

    prompts = [build_prompt(tokenizer, r["messages"][0]["content"]) for r in sample]

    print("base model:")
    with model.disable_adapter():
        base_out = generate(model, tokenizer, prompts, args.max_new_tokens, args.batch)
    print("lora model:")
    lora_out = generate(model, tokenizer, prompts, args.max_new_tokens, args.batch)

    out_path = os.path.join(args.run, "generations.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r, b, l in zip(sample, base_out, lora_out):
            for name, resp in (("base", b), ("lora", l)):
                f.write(json.dumps({"scenario_id": r["scenario_id"], "condition": r["condition"],
                                    "model": name, "prompt": r["messages"][0]["content"],
                                    "response": resp}, ensure_ascii=False) + "\n")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
