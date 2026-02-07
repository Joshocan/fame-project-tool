#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fame.config.load import load_config
from fame.rag.ss_pipeline import SSRGFMConfig, run_ss_rgfm
from fame.loggers import get_logger, log_exception
from fame.exceptions import UserMessageError, format_error


def main() -> None:
    cfg_yaml = load_config()
    p_cfg = cfg_yaml.pipelines.ss_nonrag  # reuse defaults where sensible

    ap = argparse.ArgumentParser(description="Run Single-Stage RAG Generated Feature Modeling (SS-RGFM)")
    ap.add_argument("--root-feature", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--prompt-path", default="", help="Custom prompt template path")
    ap.add_argument("--n-results-per-collection", type=int, default=6)
    ap.add_argument("--max-total-results", type=int, default=12)
    ap.add_argument("--max-total-chars", type=int, default=18_000)
    ap.add_argument("--max-chunk-chars", type=int, default=2_500)
    ap.add_argument("--collection-mode", default="per_source", choices=["per_source", "one_collection"])
    ap.add_argument("--one-collection-name", default="fame_all")
    ap.add_argument("--collection-prefix", default="")
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()

    chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else None
    prompt_path = Path(args.prompt_path).expanduser() if args.prompt_path else None

    cfg = SSRGFMConfig(
        root_feature=args.root_feature,
        domain=args.domain,
        chunks_dir=chunks_dir,
        chunks_files=None,
        collection_mode=args.collection_mode,
        one_collection_name=args.one_collection_name,
        collection_prefix=args.collection_prefix,
        batch_size=args.batch_size,
        n_results_per_collection=args.n_results_per_collection,
        max_total_results=args.max_total_results,
        max_total_chars=args.max_total_chars,
        max_chunk_chars=args.max_chunk_chars,
        prompt_path=prompt_path,
        temperature=args.temperature,
    )

    out = run_ss_rgfm(cfg)
    print("✅ SS-RGFM completed")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    logger = get_logger("ss_rgfm")
    try:
        main()
    except UserMessageError as e:
        print(f"❌ {format_error(e)} (see results/logs/fame.log for details)")
        log_exception(logger, e)
    except Exception as e:
        log_exception(logger, e)
        raise
