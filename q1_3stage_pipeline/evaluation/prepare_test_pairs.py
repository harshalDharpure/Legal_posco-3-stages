#!/usr/bin/env python3
"""
Flatten dialogue-level test JSONL into pair-level rows for eval + generation.

Each output line has:
  - prompt, output (reference), statutes_cited, dialogue_id, turn_index

Same rolling-window contract as DatasetBuilder / training.
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

from q1_3stage_pipeline.utils import DatasetBuilder, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dialogue-jsonl",
        default="q1_3stage_pipeline/data/test_20_dialogues.jsonl",
        help="Dialogue-level JSONL (with `turns`).",
    )
    ap.add_argument(
        "--out-jsonl",
        default="q1_3stage_pipeline/logs/eval_cache/test_pairs_flat.jsonl",
        help="Pair-level JSONL for generate_preds + run_eval.",
    )
    args = ap.parse_args()

    path = args.dialogue_jsonl if os.path.isabs(args.dialogue_jsonl) else str(_REPO / args.dialogue_jsonl)
    out_path = args.out_jsonl if os.path.isabs(args.out_jsonl) else str(_REPO / args.out_jsonl)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = load_jsonl(path)
    flat = DatasetBuilder(rows).build_sft()
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in flat:
            row = {
                "dialogue_id": ex.get("dialogue_id", ""),
                "turn_index": ex.get("turn_index", 0),
                "prompt": ex["prompt"],
                "output": ex["output"],
                "statutes_cited": list(ex.get("statutes_cited", []) or []),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(flat)} pairs to {out_path}")


if __name__ == "__main__":
    main()
