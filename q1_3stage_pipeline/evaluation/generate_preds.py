#!/usr/bin/env python3
"""
Greedy generation for each `prompt` in a pair-level JSONL; writes one JSON object
per line with key `candidate` (same order as input) for run_eval.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import AutoPeftModelForCausalLM
    except ImportError as e:
        raise SystemExit(f"Install torch transformers peft: {e}") from e

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF folder: full model or PEFT adapter (with adapter_config.json).")
    ap.add_argument("--pairs-jsonl", required=True, help="Flat pairs from prepare_test_pairs.py (must include `prompt`).")
    ap.add_argument("--out-jsonl", required=True, help="Predictions JSONL (one {\"candidate\": ...} per line).")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    model_path = args.model_path if os.path.isabs(args.model_path) else str(_REPO / args.model_path)
    pairs_path = args.pairs_jsonl if os.path.isabs(args.pairs_jsonl) else str(_REPO / args.pairs_jsonl)
    out_path = args.out_jsonl if os.path.isabs(args.out_jsonl) else str(_REPO / args.out_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows: list[dict] = []
    with open(pairs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = None
    if args.load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    is_adapter = os.path.isfile(os.path.join(model_path, "adapter_config.json"))
    if is_adapter:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb,
        )
    model.eval()
    device = next(model.parameters()).device

    with open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                out.write(json.dumps({"candidate": ""}, ensure_ascii=False) + "\n")
                continue
            inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            # Decode only new tokens
            in_len = inputs["input_ids"].shape[1]
            new_tokens = gen_ids[0, in_len:]
            text = tok.decode(new_tokens, skip_special_tokens=True).strip()
            out.write(json.dumps({"candidate": text}, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{len(rows)}", flush=True)

    print(f"Wrote {len(rows)} predictions to {out_path}")


if __name__ == "__main__":
    main()
