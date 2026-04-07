#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import re
from pathlib import Path

from fame.evaluation import analyze_prompt_usage
from fame.context import ContextBuildConfig, ContextManager, chunks_from_chunks_json
from fame.nonrag.ss_pipeline import SSNonRagConfig
from fame.nonrag.prompt_utils import build_ss_nonrag_prompt, save_modified_prompt
from fame.utils.dirs import build_paths, ensure_dir
from fame.utils.context_budget import compute_max_total_chars, compute_max_chunks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SS-NonRAG context usage for a given model.")
    p.add_argument("--model", required=True, help="LLM model name (e.g., gpt-oss:120b-cloud)")
    p.add_argument("--prompt-file", default="", help="Path to modified prompt file. Defaults to latest in results.")
    p.add_argument("--root-feature", default="Model Federation")
    p.add_argument("--domain", default="Model Driven Engineering")
    p.add_argument("--inputfile", default="", help="Optional placeholder value for inputfile")
    p.add_argument("--mmdir", default="", help="Optional placeholder value for mmdir")
    return p.parse_args()


def _find_latest_prompt(prompt_dir: Path) -> Path:
    files = sorted(prompt_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No prompt files found in {prompt_dir}")
    return files[-1]


def main() -> None:
    args = parse_args()

    paths = build_paths()
    ensure_dir(paths.modified_prompts)
    out_dir = paths.non_ss_root
    ensure_dir(out_dir)

    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        prompt_text = prompt_path.read_text(encoding="utf-8")
    else:
        # Build prompt without running the LLM
        chunks_dir = paths.processed_data / "chunks"
        ensure_dir(chunks_dir)
        files = sorted(chunks_dir.glob("*.chunks.json"))
        if not files:
            raise FileNotFoundError(f"No chunks.json found in {chunks_dir}. Run ingestion first.")

        all_chunks = []
        for f in files:
            all_chunks.extend(chunks_from_chunks_json(f))
        if not all_chunks:
            raise RuntimeError(f"Chunks exist but are empty in {chunks_dir}.")

        cm = ContextManager()
        max_total_chars = compute_max_total_chars(args.model) or 140_000
        max_chunks = compute_max_chunks(max_total_chars=max_total_chars)
        ctx_cfg = ContextBuildConfig(
            max_total_chars=max_total_chars,
            max_chunks=max_chunks,
            max_chunk_chars=6_000,
            include_headers=True,
            order="by_page_then_id",
        )
        context = cm.add_initial_context(all_chunks, ctx_cfg, title="IN-LEARNING CONTEXT (SS-NONRAG)")

        cfg = SSNonRagConfig(
            root_feature=args.root_feature,
            domain=args.domain,
        )
        prompt_text = build_ss_nonrag_prompt(
            cfg,
            context=context,
            paths=paths,
            extra_placeholders={"inputfile": args.inputfile, "mmdir": args.mmdir},
            strict=False,
        )

        ts = time.strftime("%Y-%m-%dT%H-%M-%S")
        model_safe = re.sub(r"[^a-zA-Z0-9]+", "-", args.model).strip("-").lower()
        prompt_path = save_modified_prompt(
            prompt=prompt_text, model_safe=model_safe, ts=ts, paths=paths, pipeline_type="ss_nonrag"
        )
    stats = analyze_prompt_usage(model=args.model, prompt_text=prompt_text)

    out_path = out_dir / f"ss_nonrag_context_stats_{args.model.replace(':', '-').replace('/', '-')}.json"
    out_path.write_text(json.dumps(stats.__dict__, indent=2), encoding="utf-8")

    print(f"SUCCESS: Context stats saved to: {out_path}")


if __name__ == "__main__":
    main()
