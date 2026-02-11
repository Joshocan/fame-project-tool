#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fame.config.load import load_config
from fame.judge import create_judge_client
from fame.nonrag.is_pipeline import ISNonRagConfig, run_is_nonrag
from fame.nonrag.cli_utils import prompt_choice, load_key_file, default_high_level_features
from fame.exceptions import MissingKeyError, UserMessageError, format_error
from fame.loggers import get_logger, log_exception


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Iterated-Stage Non-RAG (IS-NonRAG)")
    ap.add_argument("--root-feature", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--max-delta-chars", type=int, default=int(os.getenv("NONRAG_DELTA_CHARS", "50000")))
    ap.add_argument("--max-delta-chunks", type=int, default=int(os.getenv("NONRAG_DELTA_CHUNKS", "50")))
    ap.add_argument("--temperature", type=float, default=float(os.getenv("NONRAG_TEMP", "0.2")))
    ap.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    ap.add_argument("--xsd-path", default="", help="Override XSD path (default: feature_model_featureide.xsd)")
    ap.add_argument("--feature-metamodel-path", default="", help="Override feature metamodel path")
    args = ap.parse_args()

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

        args.domain = domain
        args.root_feature = root_feature

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
        args.high_level_features = high_level_features

    chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else None

    cfg_default = load_config().pipelines.is_nonrag

    cfg = ISNonRagConfig(
        root_feature=args.root_feature,
        domain=args.domain,
        chunks_dir=chunks_dir,
        max_delta_chars=args.max_delta_chars,
        max_delta_chunks=args.max_delta_chunks,
        temperature=args.temperature,
        high_level_features=getattr(args, "high_level_features", None),
        initial_prompt_path=cfg_default.initial_prompt_path,
        iter_prompt_path=cfg_default.iter_prompt_path,
        xsd_path=Path(args.xsd_path).expanduser().resolve() if args.xsd_path else None,
        feature_metamodel_path=Path(args.feature_metamodel_path).expanduser().resolve() if args.feature_metamodel_path else None,
    )

    model_name = getattr(llm_client, "model", None) or os.getenv("OLLAMA_LLM_MODEL", "ollama-default")
    print("\n==================== IS-NONRAG ====================")
    print(f"Root feature   : {cfg.root_feature}")
    print(f"Domain         : {cfg.domain}")
    print(f"Model          : {model_name}")
    print(f"Chunks dir     : {chunks_dir or '(default)'}")
    print(f"Max delta chars: {cfg.max_delta_chars}")
    print(f"Max delta chunks: {cfg.max_delta_chunks}")
    print("---------------------------------------------------")
    print("Stage 1: Build configuration")
    print("Stage 2: Run IS-NonRAG pipeline (iterative; may take a while)...")

    out = run_is_nonrag(cfg, llm=llm_client)
    print("\nSUCCESS: IS-NonRAG completed")
    print(out)


if __name__ == "__main__":
    logger = get_logger("is_nonrag")
    try:
        main()
    except UserMessageError as e:
        print(f"ERROR: {format_error(e)} (see results/logs/fame.log for details)")
        log_exception(logger, e)
    except Exception as e:
        log_exception(logger, e)
        raise
