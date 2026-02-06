#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fame.config.load import load_config
from fame.judge import create_judge_client
from fame.nonrag.is_pipeline import ISNonRagConfig, run_is_nonrag
from fame.nonrag.cli_utils import prompt_choice, load_key_file, default_high_level_features


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Iterated-Stage Non-RAG (IS-NonRAG)")
    ap.add_argument("--root-feature", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--max-delta-chars", type=int, default=int(os.getenv("NONRAG_DELTA_CHARS", "50000")))
    ap.add_argument("--max-delta-chunks", type=int, default=int(os.getenv("NONRAG_DELTA_CHUNKS", "50")))
    ap.add_argument("--temperature", type=float, default=float(os.getenv("NONRAG_TEMP", "0.2")))
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
                raise RuntimeError(f"Missing API key file: {key_file}")
            os.environ[env_var] = key

            cfg_judge = load_config().llm_judge
            llm_client = create_judge_client(
                provider=provider,
                model=model,
                base_url=cfg_judge.base_url,
                api_key_env=env_var,
                temperature=cfg_judge.temperature,
                max_tokens=cfg_judge.max_tokens,
                timeout_s=cfg_judge.timeout_s,
            )

        domain = input("Enter domain [Model Driven Engineering]: ").strip() or "Model Driven Engineering"
        root_feature = input("Enter root feature [Model Federation]: ").strip() or "Model Federation"

        args.domain = domain
        args.root_feature = root_feature

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
        args.high_level_features = high_level_features

    chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else None

    cfg = ISNonRagConfig(
        root_feature=args.root_feature,
        domain=args.domain,
        chunks_dir=chunks_dir,
        max_delta_chars=args.max_delta_chars,
        max_delta_chunks=args.max_delta_chunks,
        temperature=args.temperature,
        high_level_features=getattr(args, "high_level_features", None),
    )

    out = run_is_nonrag(cfg, llm=llm_client)
    print("\n✅ IS-NonRAG completed")
    print(out)


if __name__ == "__main__":
    main()
