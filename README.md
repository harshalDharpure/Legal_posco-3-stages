# Legal POSCO — 3-stage dialogue research pipeline

This repository contains the strict **Stage 1 (SFT) → Stage 2 (multi-objective) → Stage 3 (DPO)** pipeline with a fixed **70/10/20** dialogue split.

## Repository layout

```
Legal_posco-3stages/
├── datasets/                    # All JSONL data (see datasets/README.md)
│   ├── dialogue_splits_70_10_20/
│   ├── merged/                  # auto-built final_train (gitignored)
│   └── scripts/
├── q1_3stage_pipeline/
│   ├── stage1/                  # SFT training
│   ├── stage2/                  # multi-objective training
│   ├── stage3/                  # DPO training
│   ├── evaluation/
│   ├── configs/
│   ├── scripts/
│   ├── ablation/
│   └── outputs/                 # checkpoints & run logs (gitignored)
├── PROJECT_DATASET_DETAILS.md
└── RESEARCH_HANDOFF.md
```

## Directory layout (pipeline code)

- `q1_3stage_pipeline/configs/`: default YAML (`pipeline_default.yaml`)
- `q1_3stage_pipeline/stage1/`: masked causal LM SFT
- `q1_3stage_pipeline/stage2/`: \(L_{gen} + \lambda_1 L_{entail} + \lambda_2 L_{triplet}\)
- `q1_3stage_pipeline/stage3/`: TRL DPO
- `q1_3stage_pipeline/evaluation/`: metrics and test eval scripts
- `q1_3stage_pipeline/outputs/`: local checkpoints (gitignored)

## Quickstart

Run from the **repo root** (so relative paths work):

All commands below assume `python3` is available (on this machine `python` may not exist).

### Prerequisites (Hugging Face access + token)

You must have access to the base model repo:
`meta-llama/Meta-Llama-3.1-8B-Instruct`.

Set your Hugging Face token **in your shell** (do not write it into any file).

```bash
export HF_TOKEN="PASTE_YOUR_TOKEN_HERE"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

Optional verification:

```bash
python3 -c "import os; print('HF_TOKEN' in os.environ, 'HUGGINGFACE_HUB_TOKEN' in os.environ)"
```

Alternative persistent login (writes to your user cache):

```bash
huggingface-cli login
```

### 0) Dataset (already in repo)

Official dialogue splits are under `datasets/dialogue_splits_70_10_20/` (see `datasets/README.md`).

```bash
python3 datasets/scripts/create_70_10_20_split_dialogue_level.py   # optional: recreate splits
```

### Dialogue-level 70/10/20 split (recommended)

If you already created the dialogue-level split (NO pairs) with:
`data/create_70_10_20_split_dialogue_level.py`,
then use:

- `datasets/dialogue_splits_70_10_20/train_70_dialogues.jsonl`
- `datasets/dialogue_splits_70_10_20/val_10_dialogues.jsonl`
- `datasets/dialogue_splits_70_10_20/test_20_dialogues.jsonl`

The training code will flatten dialogues into (input, output) examples **in-memory**.

## STRICT experiment protocol (no leakage)

- **train / validation / test** splits are strict:
  - Train ONLY on train
  - Validation ONLY for tuning (hyperparams / early stopping / selecting β)
  - Test NEVER used until the very end
- After tuning is complete:
  - Create **final_train = train + validation**
  - Retrain **Stage 2 (M2)** and **Stage 3 (M3)** from scratch using `final_train`
  - Run evaluation **exactly once** on test

## Global formatting contract (must stay consistent)

All stages use the same strict prompt template:

```text
[USER]: {input}
[ASSISTANT]:
```

### 2) Stage 1 — SFT (M1)

```bash
python3 q1_3stage_pipeline/stage1/train.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --train-jsonl datasets/dialogue_splits_70_10_20/train_70_dialogues.jsonl \
  --val-jsonl datasets/dialogue_splits_70_10_20/val_10_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/outputs/checkpoints/stage1/M1_seed42 \
  --seed 42
```

### 3) Stage 2 — Multi-objective (M2)

Stage 2 is **initialized from M1** and trains:

\[
L = L_{gen} + \lambda_1 L_{entail} + \lambda_2 L_{triplet}
\]

- \(L_{entail}\): frozen DeBERTa-large MNLI teacher + KL distillation head (teacher forcing; no gradient through decoding)
- \(L_{triplet}\): dynamic hard negatives (model-gen + legal corruption + cross-sample) + SBERT filtering + hard mining

```bash
python3 q1_3stage_pipeline/stage2/train.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --init-from m1 \
  --m1-path q1_3stage_pipeline/outputs/checkpoints/stage1/M1_seed42/final \
  --ablation full \
  --train-jsonl datasets/dialogue_splits_70_10_20/train_70_dialogues.jsonl \
  --val-jsonl datasets/dialogue_splits_70_10_20/val_10_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/outputs/checkpoints/stage2/M2_fromM1_full_seed42 \
  --eval-every 50 \
  --seed 42
```

### 4) Stage 3 — DPO (M3)

Stage 3 runs DPO where:
- **chosen** = ground truth
- **rejected** = dynamic hard negatives (generated on-the-fly; not stored in the dataset)
- **reference model** = M2 (frozen)

```bash
python3 q1_3stage_pipeline/stage3/train.py \
  --m2-path q1_3stage_pipeline/outputs/checkpoints/stage2/M2_fromM1_full_seed42/final \
  --train-jsonl datasets/dialogue_splits_70_10_20/train_70_dialogues.jsonl \
  --output-dir q1_3stage_pipeline/outputs/checkpoints/stage3/M3_beta0.1_seed42 \
  --beta 0.1 \
  --seed 42
```

#### β sweep (required)

```bash
for beta in 0.1 0.5 1.0; do
  python3 q1_3stage_pipeline/stage3/train.py \
    --m2-path q1_3stage_pipeline/outputs/checkpoints/stage2/M2_fromM1_full_seed42/final \
    --train-jsonl datasets/dialogue_splits_70_10_20/train_70_dialogues.jsonl \
    --output-dir "q1_3stage_pipeline/outputs/checkpoints/stage3/M3_beta${beta}_seed42" \
    --beta "$beta" \
    --seed 42
done
```

### 5) Stage 2 ablations

```bash
python3 q1_3stage_pipeline/ablation/run_stage2_ablations.py \
  --config q1_3stage_pipeline/configs/pipeline_default.yaml \
  --train-jsonl datasets/dialogue_splits_70_10_20/train_70_dialogues.jsonl \
  --val-jsonl datasets/dialogue_splits_70_10_20/val_10_dialogues.jsonl \
  --m1-path q1_3stage_pipeline/outputs/checkpoints/stage1/M1_seed42/final \
  --out-root q1_3stage_pipeline/outputs/checkpoints/stage2_ablations
```

### 6) Evaluation helper (reference/candidate pairs)

You must generate model outputs on the **test split** first, then run `run_eval.py`.
`pred-jsonl` must be in the same order as `test-jsonl` and contain either `candidate` or `output`.

```bash
python3 q1_3stage_pipeline/evaluation/run_eval.py \
  --test-jsonl datasets/dialogue_splits_70_10_20/test_20_dialogues.jsonl \
  --pred-jsonl q1_3stage_pipeline/outputs/preds.jsonl
```

`run_eval.py` reports automatic metrics (ROUGE/BLEU/METEOR/NLI) plus:
- statute correctness proxies from `statutes_cited`
- safety/refusal proxy rates

## Multiple seeds (required)

Run every reported setting with **3 random seeds** (example: 42, 43, 44) and report mean/std.

Example for Stage 1:

```bash
for seed in 42 43 44; do
  python3 q1_3stage_pipeline/stage1/train.py \
    --config q1_3stage_pipeline/configs/pipeline_default.yaml \
    --train-jsonl datasets/dialogue_splits_70_10_20/train_70_dialogues.jsonl \
    --val-jsonl datasets/dialogue_splits_70_10_20/val_10_dialogues.jsonl \
    --output-dir "q1_3stage_pipeline/outputs/checkpoints/stage1/M1_seed${seed}" \
    --seed "$seed"
done
```

