"""Data loading, train/test splitting and tokenisation.

Expected input: a JSONL file where each line looks like
    {"messages": [{"role": "user", "content": ...},
                  {"role": "assistant", "content": ...}],
     "scenario_id": int, "condition": int, "split_group": int | str, ...}
"""
import json
import random

from datasets import Dataset


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_rows(rows, test_frac=0.2, seed=42, mode="group"):
    """Return (train_rows, test_rows).

    mode="group":  hold out whole request groups, so the same request never
                   appears in both splits (measures generalisation to unseen requests).
    mode="random": plain random row split (faster to set up, but test requests
                   also occur in training under different narratives).
    """
    rng = random.Random(seed)

    if mode == "random":
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        test_idx = set(idx[: int(len(rows) * test_frac)])
        return ([r for i, r in enumerate(rows) if i not in test_idx],
                [r for i, r in enumerate(rows) if i in test_idx])

    groups = sorted({str(r["split_group"]) for r in rows if str(r["split_group"]) != "overt"})
    rng.shuffle(groups)
    test_groups = set(groups[: max(1, int(len(groups) * test_frac))])

    overt = sorted({r["scenario_id"] for r in rows if str(r["split_group"]) == "overt"})
    rng.shuffle(overt)
    test_overt = set(overt[: int(len(overt) * test_frac)])

    def is_test(r):
        if str(r["split_group"]) == "overt":
            return r["scenario_id"] in test_overt
        return str(r["split_group"]) in test_groups

    return [r for r in rows if not is_test(r)], [r for r in rows if is_test(r)]


def build_prompt(tokenizer, user_text):
    """Chat-formatted prompt ending where the assistant starts speaking."""
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try: 
        return tokenizer.apply_chat_template([{"role": "user", "content": user_text}],
                                             enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template([{"role": "user", "content": user_text}], **kwargs)


def encode(example, tokenizer, max_len=1024):
    """input_ids = prompt + answer; labels mask the prompt with -100 (loss on answer only)."""
    prompt_ids = tokenizer(build_prompt(tokenizer, example["messages"][0]["content"]),
                           add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example["messages"][1]["content"] + tokenizer.eos_token,
                           add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    return {"input_ids": input_ids, "labels": labels}


def to_dataset(rows, tokenizer, max_len=1024):
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])
    return ds.map(lambda ex: encode(ex, tokenizer, max_len), remove_columns=["messages"])
