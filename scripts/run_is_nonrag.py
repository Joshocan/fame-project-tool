# scripts/run_is_nonrag.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fame.nonrag.is_pipeline import ISNonRagConfig, run_is_nonrag


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Iterated-Stage Non-RAG (IS-NonRAG)")
    ap.add_argument("--root-feature", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--max-delta-chars", type=int, default=int(os.getenv("NONRAG_DELTA_CHARS", "50000")))
    ap.add_argument("--max-delta-chunks", type=int, default=int(os.getenv("NONRAG_DELTA_CHUNKS", "50")))
    ap.add_argument("--temperature", type=float, default=float(os.getenv("NONRAG_TEMP", "0.2")))
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else None

    cfg = ISNonRagConfig(
        root_feature=args.root_feature,
        domain=args.domain,
        chunks_dir=chunks_dir,
        max_delta_chars=args.max_delta_chars,
        max_delta_chunks=args.max_delta_chunks,
        temperature=args.temperature,
    )

    out = run_is_nonrag(cfg)
    print("\n✅ IS-NonRAG completed")
    print(out)


if __name__ == "__main__":
    main()
