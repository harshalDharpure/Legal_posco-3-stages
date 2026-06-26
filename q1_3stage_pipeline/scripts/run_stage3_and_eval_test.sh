#!/usr/bin/env bash
# Run Stage 3 (DPO from M2) then test-set generation + evaluation.
# Requires: completed Stage 1 + Stage 2 (M2 checkpoint), HF token for Llama.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   export HF_TOKEN=...  HUGGINGFACE_HUB_TOKEN=...
#   bash q1_3stage_pipeline/scripts/run_stage3_and_eval_test.sh
#
# Override defaults:
#   M2_PATH=... M3_OUT=... TRAIN_JSONL=... bash q1_3stage_pipeline/scripts/run_stage3_and_eval_test.sh
#
# Skip DPO if M3 final already exists:
#   SKIP_DPO=1 bash q1_3stage_pipeline/scripts/run_stage3_and_eval_test.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

M2_PATH="${M2_PATH:-q1_3stage_pipeline/outputs/checkpoints/stage2/M2_fromM1_seed43_full_resume_gpu4/best}"
M3_OUT="${M3_OUT:-q1_3stage_pipeline/outputs/checkpoints/stage3/M3_eval_run}"
# If set, skip DPO and generate from this checkpoint (e.g. M2 best for baseline).
GEN_MODEL_PATH="${GEN_MODEL_PATH:-}"
TRAIN_JSONL="${TRAIN_JSONL:-datasets/dialogue_splits_70_10_20/train_70_dialogues.jsonl}"
TEST_DIALOGUES="${TEST_DIALOGUES:-datasets/dialogue_splits_70_10_20/test_20_dialogues.jsonl}"
PAIRS="${PAIRS:-q1_3stage_pipeline/outputs/eval_cache/test_pairs_flat.jsonl}"
PREDS="${PREDS:-q1_3stage_pipeline/outputs/eval_cache/test_preds.jsonl}"
METRICS="${METRICS:-q1_3stage_pipeline/outputs/eval_cache/test_metrics.json}"
MIN_NLI="${MIN_NLI:-0.70}"
BETA="${BETA:-0.1}"
SEED="${SEED:-43}"
LOAD_4BIT="${LOAD_4BIT:-1}"

mkdir -p q1_3stage_pipeline/outputs/eval_cache

echo "[1/4] Flatten test dialogues -> pairs"
python3 -u q1_3stage_pipeline/evaluation/prepare_test_pairs.py \
  --dialogue-jsonl "$TEST_DIALOGUES" \
  --out-jsonl "$PAIRS"

if [[ "${SKIP_DPO:-0}" != "1" && -z "${GEN_MODEL_PATH}" ]]; then
  echo "[2/4] Stage 3 DPO (from M2)"
  S3_CMD=(python3 -u q1_3stage_pipeline/stage3/train.py
    --m2-path "$M2_PATH"
    --train-jsonl "$TRAIN_JSONL"
    --output-dir "$M3_OUT"
    --beta "$BETA"
    --seed "$SEED"
    --epochs 1.0
    --lr 5e-6
    --batch-size 1
    --grad-accum 8)
  if [[ "$LOAD_4BIT" == "1" ]]; then
    S3_CMD+=(--load-in-4bit)
  fi
  "${S3_CMD[@]}"
else
  if [[ -n "${GEN_MODEL_PATH}" ]]; then
    echo "[2/4] Skipping DPO — using GEN_MODEL_PATH=$GEN_MODEL_PATH for generation"
  else
    echo "[2/4] SKIP_DPO=1 — using existing M3 at $M3_OUT/final"
  fi
fi

if [[ -n "${GEN_MODEL_PATH}" ]]; then
  M3_FINAL="$GEN_MODEL_PATH"
else
  M3_FINAL="${M3_OUT}/final"
fi
if [[ ! -d "$M3_FINAL" ]]; then
  echo "ERROR: M3 final not found: $M3_FINAL" >&2
  exit 1
fi

echo "[3/4] Generate test predictions (greedy)"
GEN_CMD=(python3 -u q1_3stage_pipeline/evaluation/generate_preds.py
  --model-path "$M3_FINAL"
  --pairs-jsonl "$PAIRS"
  --out-jsonl "$PREDS")
if [[ "$LOAD_4BIT" == "1" ]]; then
  GEN_CMD+=(--load-in-4bit)
fi
"${GEN_CMD[@]}"

echo "[4/4] Evaluate (require mean NLI entailment >= $MIN_NLI)"
python3 -u q1_3stage_pipeline/evaluation/run_eval.py \
  --test-jsonl "$PAIRS" \
  --pred-jsonl "$PREDS" \
  --min-nli "$MIN_NLI" \
  --metrics-json "$METRICS"

echo "Done. Metrics: $METRICS"
