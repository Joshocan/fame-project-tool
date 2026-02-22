#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fame.config.load import load_config
from fame.rag.is_pipeline import ISRgfmConfig, run_is_rgfm
from fame.loggers import get_logger, log_exception
from fame.exceptions import UserMessageError, MissingKeyError, format_error
from fame.nonrag.cli_utils import prompt_choice, load_key_file, default_high_level_features
from fame.judge import create_judge_client


def main() -> None:
    cfg_yaml = load_config()

    ap = argparse.ArgumentParser(description="Run Iterative RAG Generated Feature Modeling (IS-RGFM)")
    ap.add_argument("--root-feature", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--n-results-per-collection", type=int, default=6)
    ap.add_argument("--max-total-results", type=int, default=12)
    ap.add_argument("--max-total-chars", type=int, default=18_000)
    ap.add_argument("--max-chunk-chars", type=int, default=2_500)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--xsd-path", default="", help="Override XSD path")
    ap.add_argument("--feature-metamodel-path", default="", help="Override feature metamodel path")
    ap.add_argument("--initial-prompt-path", default="", help="Override initial prompt path")
    ap.add_argument("--iter-prompt-path", default="", help="Override iterative prompt path")
    ap.add_argument("--interactive", action="store_true", help="Run with guided prompts")
    ap.add_argument("--repeats", type=int, default=1, help="How many runs to execute sequentially (default: 1).")
    args = ap.parse_args()

    interactive = args.interactive or not (args.root_feature and args.domain)

    llm_client = None
    high_level_features = None

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
            judge_cfg = cfg_yaml.llm_judge
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

        hl = input("Include high-level features? (Y/n): ").strip().lower()
        feats = default_high_level_features()
        if hl not in ("n", "no"):
            print("\nHigh-level features (default):")
            for k, v in feats.items():
                print(f"- {k}: {v}")
            confirm = input("Use these? (Y/n): ").strip().lower()
            if confirm in ("n", "no"):
                high_level_features = None
            else:
                high_level_features = feats

    chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else None
    initial_prompt_path = Path(args.initial_prompt_path).expanduser() if args.initial_prompt_path else cfg_yaml.pipelines.is_rgfm_initial_prompt_path
    iter_prompt_path = Path(args.iter_prompt_path).expanduser() if args.iter_prompt_path else cfg_yaml.pipelines.is_rgfm_iter_prompt_path
    xsd_path = Path(args.xsd_path).expanduser() if args.xsd_path else None
    feature_metamodel_path = Path(args.feature_metamodel_path).expanduser() if args.feature_metamodel_path else None

    cfg = ISRgfmConfig(
        root_feature=args.root_feature,
        domain=args.domain,
        chunks_dir=chunks_dir,
        initial_prompt_path=initial_prompt_path,
        iter_prompt_path=iter_prompt_path,
        n_results_per_collection=args.n_results_per_collection,
        max_total_results=args.max_total_results,
        max_total_chars=args.max_total_chars,
        max_chunk_chars=args.max_chunk_chars,
        xsd_path=xsd_path,
        feature_metamodel_path=feature_metamodel_path,
        high_level_features=high_level_features,
        temperature=args.temperature,
    )

    print("\n==================== IS-RGFM ====================")
    print(f"Root feature   : {cfg.root_feature}")
    print(f"Domain         : {cfg.domain}")
    print(f"Model          : {(getattr(llm_client, 'model', None) or os.getenv('OLLAMA_LLM_MODEL', 'ollama-default'))}")
    print(f"Chunk server   : {chunks_dir or 'default processed_data/chunks'}")
    print("-------------------------------------------------")
    print("Stage 1: Build configuration")
    print("Stage 2: Run IS-RGFM pipeline (iterative retrieval)...")

    results = []
    for i in range(max(1, args.repeats)):
        if args.repeats > 1:
            print(f"--- Run {i+1}/{args.repeats} ---")
        out = run_is_rgfm(cfg, llm=llm_client)
        results.append(out)
        print("SUCCESS: IS-RGFM completed")
        for k, v in out.items():
            print(f"{k}: {v}")

    if args.repeats > 1:
        print(f"Completed {args.repeats} runs.")


if __name__ == "__main__":
    logger = get_logger("is_rgfm")
    try:
        main()
    except UserMessageError as e:
        print(f"ERROR: {format_error(e)} (see results/logs/fame.log for details)")
        log_exception(logger, e)
    except Exception as e:
        log_exception(logger, e)
        raise
