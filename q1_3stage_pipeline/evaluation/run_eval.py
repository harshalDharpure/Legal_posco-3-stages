#!/usr/bin/env python3
"""
Run automatic metrics (ROUGE, BLEU, METEOR, NLI) on reference/candidate pairs.

Examples:
  python q1_3stage_pipeline/evaluation/run_eval.py --refs-cands q1_3stage_pipeline/outputs/sample_pairs.json

  python q1_3stage_pipeline/evaluation/run_eval.py \
    --test-jsonl datasets/dialogue_splits_70_10_20/test_20_dialogues.jsonl \
    --pred-jsonl q1_3stage_pipeline/outputs/preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs-cands", help="JSON list of {reference, candidate}")
    ap.add_argument("--test-jsonl", help="JSONL with input, output; requires --pred-jsonl")
    ap.add_argument("--pred-jsonl", help="Same order as test-jsonl, field candidate or output")
    ap.add_argument(
        "--min-nli",
        type=float,
        default=None,
        help="If set, exit with code 2 when mean NLI entailment score is below this threshold.",
    )
    ap.add_argument(
        "--metrics-json",
        default="",
        help="If set, also write full metrics dict to this JSON file.",
    )
    args = ap.parse_args()

    from q1_3stage_pipeline.evaluation.legal_metrics import statute_correctness_score
    from q1_3stage_pipeline.evaluation.metrics import calculate_batch_metrics, calculate_nli_score
    from q1_3stage_pipeline.evaluation.safety_metrics import harmful_output_flag, refusal_flag

    refs, cands = [], []
    statutes = []
    if args.refs_cands:
        with open(args.refs_cands, encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            refs.append(row["reference"])
            cands.append(row["candidate"])
            statutes.append(row.get("statutes_cited", []))
    elif args.test_jsonl and args.pred_jsonl:
        import json as js

        def load_jl(p):
            out = []
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.append(js.loads(line))
            return out

        tg = load_jl(args.test_jsonl)
        pr = load_jl(args.pred_jsonl)
        for a, b in zip(tg, pr):
            refs.append(a.get("output", ""))
            cands.append(b.get("candidate", b.get("output", "")))
            statutes.append(a.get("statutes_cited", []))
    else:
        raise SystemExit("Provide --refs-cands or --test-jsonl and --pred-jsonl")

    m = calculate_batch_metrics(refs, cands, lang="en")
    try:
        m.update(calculate_nli_score(refs, cands))
    except Exception as e:
        m["nli_error"] = str(e)

    # Legal + safety proxies
    legal_scores = [statute_correctness_score(statutes_cited=s, candidate=c) for s, c in zip(statutes, cands)]
    if legal_scores:
        m["statute_precision"] = sum(x["statute_precision"] for x in legal_scores) / len(legal_scores)
        m["statute_recall"] = sum(x["statute_recall"] for x in legal_scores) / len(legal_scores)
        m["statute_f1"] = sum(x["statute_f1"] for x in legal_scores) / len(legal_scores)
    m["harmful_rate"] = sum(harmful_output_flag(c) for c in cands) / max(len(cands), 1)
    m["refusal_rate"] = sum(refusal_flag(c) for c in cands) / max(len(cands), 1)

    print(json.dumps(m, indent=2))

    if args.metrics_json:
        out_p = Path(args.metrics_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(m, indent=2), encoding="utf-8")

    if args.min_nli is not None:
        nli = float(m.get("nli_score", 0.0))
        if m.get("nli_error"):
            print(f"NLI not computed: {m.get('nli_error')}", file=sys.stderr)
            sys.exit(1)
        if nli < float(args.min_nli):
            print(
                f"NLI score {nli:.4f} is below required minimum {float(args.min_nli):.4f}. "
                "Try a stronger checkpoint, more DPO training, or different beta / decoding.",
                file=sys.stderr,
            )
            sys.exit(2)


if __name__ == "__main__":
    main()

