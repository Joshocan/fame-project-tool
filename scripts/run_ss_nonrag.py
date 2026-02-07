#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

from fame.judge import create_judge_client
from fame.config.load import load_config
from fame.exceptions import MissingKeyError, UserMessageError, format_error
from fame.nonrag.ss_pipeline import SSNonRagConfig, run_ss_nonrag
from fame.nonrag.cli_utils import prompt_choice, load_key_file, default_high_level_features


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Single-Stage Non-RAG (SS-NonRAG)")
    ap.add_argument("--root-feature", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--max-total-chars", type=int, default=int(os.getenv("NONRAG_MAX_CHARS", "140000")))
    ap.add_argument("--max-chunks", type=int, default=int(os.getenv("NONRAG_MAX_CHUNKS", "120")))
    ap.add_argument("--max-chunk-chars", type=int, default=int(os.getenv("NONRAG_MAX_CHUNK_CHARS", "6000")))
    ap.add_argument("--prompt-path", default="", help="Optional prompt file path")
    ap.add_argument("--run-tag", default=os.getenv("NONRAG_RUN_TAG", "ss-nonrag"))
    ap.add_argument("--verbose", action="store_true", help="Print stage-by-stage progress")
    ap.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = ap.parse_args()

    interactive = args.interactive or not (args.root_feature and args.domain)

    llm_client = None
    if interactive:
        mode = prompt_choice("1) Open Source LLM  OR Judge LLM", ("Open Source LLM", "Judge LLM"))

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
                print("⚠️  ollama_key not found. Using local Ollama for LLM.")
                os.environ.setdefault("OLLAMA_LLM_HOST", "http://127.0.0.1:11434")

            os.environ.setdefault("OLLAMA_EMBED_HOST", "http://127.0.0.1:11434")

        else:
            model = prompt_choice(
                "Select Judge LLM model",
                ("gpt-5", "claude-opus", "gemini-3"),
            )
            provider_map = {
                "gpt-5": ("openai", "OPENAI_API_KEY", Path("api_keys/openai_key.txt")),
                "claude-opus": ("anthropic", "ANTHROPIC_API_KEY", Path("api_keys/anthropic_key.txt")),
                "gemini-3": ("gemini", "GEMINI_API_KEY", Path("api_keys/gemini_key.txt")),
            }
            provider, env_var, key_file = provider_map[model]
            key = load_key_file(key_file)
            if not key:
                raise MissingKeyError(env_var, str(key_file))
            os.environ[env_var] = key

            cfg = load_config().llm_judge
            llm_client = create_judge_client(
                provider=provider,
                model=model,
                base_url=cfg.base_url,
                api_key_env=env_var,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                timeout_s=cfg.timeout_s,
            )

        domain = input("Enter domain [Model Driven Engineering]: ").strip() or "Model Driven Engineering"
        root_feature = input("Enter root feature [Model Federation]: ").strip() or "Model Federation"

        high_level = input("Include high-level features? (y/N): ").strip().lower()
        high_level_features = None
        if high_level in ("y", "yes"):
            features = default_high_level_features()
            print("\nHigh-level features:")
            for k, v in features.items():
                print(f"- {k}: {v}")
            confirm = input("Use these? (Y/n): ").strip().lower()
            if confirm in ("", "y", "yes"):
                high_level_features = features

        args.domain = domain
        args.root_feature = root_feature
        args.high_level_features = high_level_features

    chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else None
    prompt_path = Path(args.prompt_path).expanduser().resolve() if args.prompt_path else None

    if args.verbose:
        print("\n==================== SS-NONRAG ====================")
        print(f"Root feature   : {args.root_feature}")
        print(f"Domain         : {args.domain}")
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
        run_tag=args.run_tag,
        high_level_features=getattr(args, "high_level_features", None),
    )

    if args.verbose:
        print("Stage 2: Execute SS-NonRAG pipeline")

    out = run_ss_nonrag(cfg, llm_client=llm_client)
    print("\n✅ SS-NonRAG completed")
    print(out)


if __name__ == "__main__":
    try:
        main()
    except UserMessageError as e:
        print(f"❌ {format_error(e)}")
    except Exception as e:
        raise
