"""Fine-tune a LoRA adapter on the distress x risk benchmark.

Example:
    python train.py --data data/sft.jsonl --lora full --out runs/full
    python train.py --data data/sft.jsonl --lora localized --layers 14-18 --out runs/loc_l16
"""
import argparse
import json
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)

from src.data import load_rows, split_rows, to_dataset

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def lora_config(kind, layers=None, r=16, alpha=32):
    common = dict(r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    if kind == "full":
        return LoraConfig(target_modules=ALL_LINEAR, **common)
    if kind == "localized":
        lo, hi = (int(x) for x in layers.split("-"))
        return LoraConfig(target_modules=ALL_LINEAR, layers_to_transform=list(range(lo, hi + 1)), **common)
    if kind == "attn":
        return LoraConfig(target_modules=["q_proj", "v_proj"], **common)
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--lora", choices=["full", "localized", "attn"], default="full")
    p.add_argument("--layers", default="14-18", help="inclusive layer range for --lora localized")
    p.add_argument("--split", choices=["group", "random"], default="group")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--grad_ckpt", action="store_true", help="enable on smaller GPUs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs/full")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(args.data)
    train_rows, test_rows = split_rows(rows, seed=args.seed, mode=args.split)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({"mode": args.split, "seed": args.seed,
                   "test": [[r["scenario_id"], r["condition"]] for r in test_rows]}, f)
    print(f"train={len(train_rows)} test={len(test_rows)} (split={args.split})")

    train_ds = to_dataset(train_rows, tokenizer, args.max_len)
    test_ds = to_dataset(test_rows, tokenizer, args.max_len)
    eval_ds = test_ds.shuffle(seed=args.seed).select(range(min(300, len(test_ds))))

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map={"": 0}, attn_implementation="sdpa")
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(args.lora, args.layers))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=100, save_total_limit=2,
        logging_steps=10,
        prediction_loss_only=True,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    trainer.train()

    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("saved adapter to", adapter_dir)


if __name__ == "__main__":
    main()
