#!/usr/bin/env python
"""Generate run_metadata.json with standardized fields for evaluations.

Usage example:
  PYTHONPATH=$(pwd) .venv/bin/python scripts/run_metadata.py \
    --pipeline-id SS-RGFM \
    --llm-model gpt-4.1 \
    --iteration-id 3 \
    --retrieval-enabled true \
    --top-k-chunks 12 \
    --prompt-type fm_extraction \
    --model-temperature 0.2 \
    --out results/rag/ss-rgfm/fm/ss-rgfm_response_gpt-4.1.run_metadata.json

If --timestamp is omitted, current UTC time is used.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_bool(val: str) -> bool:
    return str(val).lower() in {"1", "true", "t", "yes", "y"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate run metadata JSON")
    ap.add_argument("--pipeline-id", required=True, help="SS-Non-RAG | Iter-Non-RAG | SS-RGFM | Iter-RGFM")
    ap.add_argument("--llm-model", required=True, help="Model name used")
    ap.add_argument("--iteration-id", type=int, default=0, help="Iteration number (use 0 for single-stage)")
    ap.add_argument("--retrieval-enabled", type=str, default="false", help="true/false")
    ap.add_argument("--top-k-chunks", type=int, default=0, help="K used in retrieval (0 if not applicable)")
    ap.add_argument("--prompt-type", required=True, help="e.g., fm_extraction, fm_iterated, ...")
    ap.add_argument("--model-temperature", type=float, default=0.0, help="Temperature used for the LLM")
    ap.add_argument("--timestamp", help="ISO timestamp (UTC); defaults to now")
    ap.add_argument("--out", required=True, help="Output JSON file path")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ts = args.timestamp or datetime.utcnow().isoformat() + "Z"
    payload = {
        "pipeline_id": args.pipeline_id,
        "llm_model": args.llm_model,
        "iteration_id": args.iteration_id,
        "retrieval_enabled": parse_bool(args.retrieval_enabled),
        "top_k_chunks": args.top_k_chunks,
        "prompt_type": args.prompt_type,
        "model_temperature": args.model_temperature,
        "timestamp": ts,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
