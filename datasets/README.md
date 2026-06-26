# Datasets

All dialogue-level JSONL files for the 3-stage pipeline live here.

## Layout

| Folder | Contents |
|--------|----------|
| `dialogue_splits_70_10_20/` | Official **70% train / 10% val / 20% test** dialogue splits (strict, no leakage). |
| `merged/` | Derived files created at runtime (e.g. `final_train_dialogues.jsonl` = train + val). |
| `scripts/` | Utilities to recreate splits from a master corpus. |

## Files in `dialogue_splits_70_10_20/`

- `train_70_dialogues.jsonl` — training dialogues (tune on val only)
- `val_10_dialogues.jsonl` — validation dialogues
- `test_20_dialogues.jsonl` — held-out test (evaluate only after tuning is frozen)

Each line is one multi-turn dialogue JSON object with a `turns` array. Training code flattens turns into `(prompt, output)` pairs in memory.

## Recreate splits

From the repo root:

```bash
python3 datasets/scripts/create_70_10_20_split_dialogue_level.py
```

Point the script at your master corpus if paths differ (see script header).

## Merged train file

The orchestrator builds `datasets/merged/final_train_dialogues.jsonl` by concatenating train + val when Stage 2/3 need the full tuning set. This file is **gitignored** (regenerated locally).
