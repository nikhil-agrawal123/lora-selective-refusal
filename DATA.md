# Data

The training data is **not published** in this repository because:

- part of the distress narratives are adapted from real public posts written by people in difficult circumstances, and
- one condition consists of overtly harmful requests, which should not be redistributed as a standalone list.

It is available on request for research use.

## Format

`train.py` expects a JSONL file with one example per line:

```json
{
  "messages": [
    {"role": "user", "content": "<distress narrative or neutral filler>\n\n<question>"},
    {"role": "assistant", "content": "<target response>"}
  ],
  "scenario_id": 0,
  "condition": 1,
  "split_group": 11,
  "target_behavior": "support_decline"
}
```

| Field | Meaning |
|---|---|
| `scenario_id` | one distress narrative; shared across its C1–C4 rows |
| `condition` | 1 distress+risky · 2 distress+benign · 3 neutral+risky · 4 distress only · 5 overt harm |
| `split_group` | the underlying request; used by `--split group` so a request never appears in both train and test |
| `target_behavior` | `support_decline`, `support_answer`, `answer`, `support_only`, `refuse_overt` |

## Target responses

| Condition | Target |
|---|---|
| C1 | acknowledgement → decline the request → direct check-in → helplines |
| C2 | the same acknowledgement → full answer → (for acute narratives) gentle check-in |
| C3 | answer only; requests asking for lethality specifics get the safety-relevant answer without the specifics |
| C4 | the same acknowledgement → support → check-in where appropriate |
| C5 | refusal → check-in → helplines |

Acknowledgements are written per narrative; the rest is composed from phrase pools, deterministically per scenario.
