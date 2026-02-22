#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from fame.judge import create_judge_client
from fame.config.load import load_config
from fame.exceptions import MissingKeyError, UserMessageError, format_error
from fame.loggers import get_logger, log_exception
from fame.nonrag.ss_pipeline import SSNonRagConfig, run_ss_nonrag, _default_chunks_dir
from fame.nonrag.cli_utils import prompt_choice, load_key_file, default_high_level_features
from fame.utils.dirs import build_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Single-Stage Non-RAG (SS-NonRAG)")
    ap.add_argument("--root-feature", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--max-total-chars", type=int, default=int(os.getenv("NONRAG_MAX_CHARS", "140000")))
    ap.add_argument("--max-chunks", type=int, default=int(os.getenv("NONRAG_MAX_CHUNKS", "120")))
    ap.add_argument("--max-chunk-chars", type=int, default=int(os.getenv("NONRAG_MAX_CHUNK_CHARS", "6000")))
    ap.add_argument("--prompt-path", default="", help="Optional prompt file path")
    ap.add_argument("--xsd-path", default="", help="Override XSD path (default: feature_model_featureide.xsd)")
    ap.add_argument("--feature-metamodel-path", default="", help="Override feature metamodel path")
    ap.add_argument("--run-tag", default=os.getenv("NONRAG_RUN_TAG", "ss-nonrag"))
    ap.add_argument("--verbose", action="store_true", help="Print stage-by-stage progress")
    ap.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    ap.add_argument("--preflight", action="store_true", help="Run fast checks (no prompts, no LLM) and exit.")
    ap.add_argument("--repeats", type=int, default=1, help="How many runs to execute sequentially (default: 1).")
    args = ap.parse_args()

    if args.preflight:
        paths = build_paths()
        chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else _default_chunks_dir(paths)

        print("=== SS-NonRAG Preflight ===")
        print(f"Chunks dir   : {chunks_dir}")
        files = sorted(chunks_dir.glob("*.chunks.json"))
        if files:
            print(f"Chunks files : {len(files)}")
        else:
            print("Chunks files : 0 (ingestion will run on first pipeline execution)")

        ollama_host = os.getenv("OLLAMA_LLM_HOST", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        try:
            r = requests.get(f"{ollama_host}/api/tags", timeout=3)
            ok = 200 <= r.status_code < 400
            print(f"Ollama host  : {ollama_host} ({'ok' if ok else f'status {r.status_code}'})")
        except Exception as e:
            print(f"Ollama host  : {ollama_host} (unreachable: {e})")

        print("Preflight done. (No LLM call made.)")
        return

    interactive = args.interactive or not (args.root_feature and args.domain)

    llm_client = None
    if interactive:
        mode = prompt_choice("1) Open Source LLM  OR Proprietary LLM", ("Open Source LLM", "Proprietary LLM"))

        if mode == "Open Source LLM":
            model = prompt_choice(
                "Select Open Source LLM model",
                ("gpt-oss:120b-cloud", "glm-4.7:cloud", "deepseek-v3.2:cloud"),
            )
            os.environ["OLLAMA_LLM_MODEL"] = model

            key_path = Path("api_keys/ollama_key.txt")
            key = load_key_file(key_path)
            if key:
                os.environ["OLLAMA_API_KEY_FILE"] = str(key_path)
                os.environ.setdefault("OLLAMA_LLM_HOST", "https://ollama.com")
            else:
                print("WARN:  ollama_key not found. Using local Ollama for LLM.")
                os.environ.setdefault("OLLAMA_LLM_HOST", "http://127.0.0.1:11434")

            os.environ.setdefault("OLLAMA_EMBED_HOST", "http://127.0.0.1:11434")

        else:
            model = prompt_choice(
                "Select Proprietary LLM model",
                ("gpt-4.1", "claude-opus-4-5", "gemini-3-pro-preview"),
            )
            provider_map = {
                "gpt-4.1": ("openai", "OPENAI_API_KEY"),
                "claude-opus-4-5": ("anthropic", "ANTHROPIC_API_KEY"),
                "gemini-3-pro-preview": ("gemini", "GEMINI_API_KEY"),
            }
            provider, env_var = provider_map[model]
            judge_cfg = load_config().llm_judge
            key_file = judge_cfg.api_key_dir / f"{provider}_key.txt"
            key = load_key_file(key_file)
            if not key:
                raise MissingKeyError(env_var, str(key_file))
            os.environ[env_var] = key

            llm_client = create_judge_client(
                provider=provider,
                model=model,
                base_url=judge_cfg.base_url,
                api_key_env=env_var,
                temperature=judge_cfg.temperature,
                max_tokens=judge_cfg.max_tokens,
                timeout_s=judge_cfg.timeout_s,
            )

        domain = input("Enter domain [Model Driven Engineering]: ").strip() or "Model Driven Engineering"
        root_feature = input("Enter root feature [Model Federation]: ").strip() or "Model Federation"

        high_level = input("Include high-level features? (Y/n): ").strip().lower()
        features = default_high_level_features()
        if high_level not in ("n", "no"):
            print("\nHigh-level features (default):")
            for k, v in features.items():
                print(f"- {k}: {v}")
            confirm = input("Use these? (Y/n): ").strip().lower()
            if confirm in ("n", "no"):
                high_level_features = None
            else:
                high_level_features = features
        else:
            high_level_features = None

        args.domain = domain
        args.root_feature = root_feature
        args.high_level_features = high_level_features

    chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else None
    prompt_path = Path(args.prompt_path).expanduser().resolve() if args.prompt_path else None

    model_name = getattr(llm_client, "model", None) or os.getenv("OLLAMA_LLM_MODEL", "ollama-default")
    if args.verbose:
        print("\n==================== SS-NONRAG ====================")
        print(f"Root feature   : {args.root_feature}")
        print(f"Domain         : {args.domain}")
        print(f"Model          : {model_name}")
        print(f"Chunks dir     : {chunks_dir or '(default)'}")
        print(f"Max total chars: {args.max_total_chars}")
        print(f"Max chunks     : {args.max_chunks}")
        print(f"Max chunk chars: {args.max_chunk_chars}")
        print(f"Prompt path    : {prompt_path or '(default)'}")
        print(f"Run tag        : {args.run_tag}")
        print("---------------------------------------------------")
        print("Stage 1: Build configuration")

    cfg = SSNonRagConfig(
        root_feature=args.root_feature,
        domain=args.domain,
        chunks_dir=chunks_dir,
        max_total_chars=args.max_total_chars,
        max_chunks=args.max_chunks,
        max_chunk_chars=args.max_chunk_chars,
        prompt_path=prompt_path,
        xsd_path=Path(args.xsd_path).expanduser().resolve() if args.xsd_path else None,
        feature_metamodel_path=Path(args.feature_metamodel_path).expanduser().resolve() if args.feature_metamodel_path else None,
        run_tag=args.run_tag,
        high_level_features=getattr(args, "high_level_features", None),
    )

    if args.verbose:
        print("Stage 2: Execute SS-NonRAG pipeline (may take a while)...")

    results = []
    for i in range(max(1, args.repeats)):
        if args.repeats > 1:
            print(f"\n--- Run {i+1}/{args.repeats} ---")
        out = run_ss_nonrag(cfg, llm_client=llm_client)
        results.append(out)
        print("\nSUCCESS: SS-NonRAG completed")
        print(out)

    if args.repeats > 1:
        print(f"\nCompleted {args.repeats} runs.")


if __name__ == "__main__":
    logger = get_logger("ss_nonrag")
    try:
        main()
    except UserMessageError as e:
        print(f"ERROR: {format_error(e)} (see results/logs/fame.log for details)")
        log_exception(logger, e)
    except Exception as e:
        log_exception(logger, e)
        raise
