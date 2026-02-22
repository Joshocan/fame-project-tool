#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fame.config.load import load_config
from fame.rag.ss_pipeline import SSRGFMConfig, run_ss_rgfm
from fame.loggers import get_logger, log_exception
from fame.exceptions import UserMessageError, MissingKeyError, format_error
from fame.nonrag.cli_utils import prompt_choice, load_key_file, default_high_level_features
from fame.judge import create_judge_client


def main() -> None:
    cfg_yaml = load_config()
    p_cfg = cfg_yaml.pipelines.ss_nonrag  # reuse defaults where sensible

    ap = argparse.ArgumentParser(description="Run Single-Stage RAG Generated Feature Modeling (SS-RGFM)")
    ap.add_argument("--root-feature", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--prompt-path", default="", help="Custom prompt template path")
    ap.add_argument("--n-results-per-collection", type=int, default=6)
    ap.add_argument("--max-total-results", type=int, default=12)
    ap.add_argument(
        "--k-strategy",
        choices=["auto", "fixed"],
        default="auto",
        help="auto = half the chunks per source (current default). fixed = use --n-results-per-collection.",
    )
    ap.add_argument("--max-total-chars", type=int, default=18_000)
    ap.add_argument("--max-chunk-chars", type=int, default=2_500)
    ap.add_argument("--collection-mode", default="per_source", choices=["per_source", "one_collection"])
    ap.add_argument("--one-collection-name", default="fame_all")
    ap.add_argument("--collection-prefix", default="")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--interactive", action="store_true", help="Run with guided prompts")
    ap.add_argument("--xsd-path", default="", help="Override XSD path (default: feature_model_featureide.xsd)")
    ap.add_argument("--feature-metamodel-path", default="", help="Override feature metamodel path")
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
    prompt_path = Path(args.prompt_path).expanduser() if args.prompt_path else None
    if prompt_path is None:
        prompt_path = cfg_yaml.pipelines.ss_rgfm_prompt_path

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
        k_strategy=args.k_strategy,
        prompt_path=prompt_path,
        temperature=args.temperature,
        xsd_path=Path(args.xsd_path).expanduser().resolve() if args.xsd_path else None,
        feature_metamodel_path=Path(args.feature_metamodel_path).expanduser().resolve() if args.feature_metamodel_path else None,
        high_level_features=high_level_features,
    )

    print("\n==================== SS-RGFM ====================")
    print(f"Root feature   : {cfg.root_feature}")
    print(f"Domain         : {cfg.domain}")
    print(f"Model          : {(getattr(llm_client, 'model', None) or os.getenv('OLLAMA_LLM_MODEL', 'ollama-default'))}")
    chroma_mode = os.getenv("CHROMA_MODE", "persistent").lower()
    if chroma_mode == "http":
        chroma_host = os.getenv("CHROMA_HOST", "127.0.0.1")
        chroma_port = os.getenv("CHROMA_PORT", "8000")
        chroma_info = f"http://{chroma_host}:{chroma_port}"
    else:
        chroma_path = os.getenv("CHROMA_PATH", "data/chroma_db")
        chroma_info = f"persistent @ {chroma_path}"
    print(f"Chunk server   : Chroma ({chroma_info}) [collections from {chunks_dir or 'default processed_data/chunks'}]")
    print("-------------------------------------------------")
    print("Stage 1: Build configuration")
    print(f"Stage 2: Run SS-RGFM pipeline (may take a while)...")

    results = []
    for i in range(max(1, args.repeats)):
        if args.repeats > 1:
            print(f"\n--- Run {i+1}/{args.repeats} ---")
        out = run_ss_rgfm(cfg, llm=llm_client)
        results.append(out)
        print("SUCCESS: SS-RGFM completed")
        for k, v in out.items():
            print(f"{k}: {v}")

    if args.repeats > 1:
        print(f"\nCompleted {args.repeats} runs.")


if __name__ == "__main__":
    logger = get_logger("ss_rgfm")
    try:
        main()
    except UserMessageError as e:
        print(f"ERROR: {format_error(e)} (see results/logs/fame.log for details)")
        log_exception(logger, e)
    except Exception as e:
        log_exception(logger, e)
        raise
