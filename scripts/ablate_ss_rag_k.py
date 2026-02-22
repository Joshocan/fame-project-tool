#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from fame.config.load import load_config
from fame.exceptions import MissingKeyError, UserMessageError, format_error
from fame.judge import create_judge_client
from fame.loggers import get_logger, log_exception
from fame.nonrag.cli_utils import default_high_level_features, load_key_file, prompt_choice
from fame.rag.ss_pipeline import SSRGFMConfig, run_ss_rgfm
from fame.utils.dirs import build_paths, ensure_dir

DEFAULT_K_VALUES = [5, 10, 20, 30]
DEFAULT_K_VALUES_ONE_COLLECTION = [80, 120, 160, 200]
FIXED_PROVIDER_TIMEOUT_S = 500
DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4.1",
    "anthropic": "claude-opus-4-5",
    "gemini": "gemini-3-pro-preview",
}
DEFAULT_KEY_ENV_BY_PROVIDER = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def _parse_k_values(raw_values: Sequence[str]) -> List[int]:
    vals: List[int] = []
    seen = set()
    for raw in raw_values:
        for part in str(raw).split(","):
            item = part.strip()
            if not item:
                continue
            try:
                k = int(item)
            except ValueError as e:
                raise argparse.ArgumentTypeError(f"Invalid k value '{item}'") from e
            if k < 1:
                raise argparse.ArgumentTypeError("k values must be >= 1")
            if k not in seen:
                vals.append(k)
                seen.add(k)
    if not vals:
        raise argparse.ArgumentTypeError("At least one k value is required")
    return vals


def _resolve_optional_path(raw: str) -> Optional[Path]:
    if not str(raw).strip():
        return None
    return Path(raw).expanduser().resolve()


def _prompt_int(label: str, *, default: int, min_value: int = 0) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            print("Invalid integer. Try again.")
            continue
        if val < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return val


def _prompt_float(label: str, *, default: float) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Try again.")


def _check_complete_feature_model_xml(xml_path: Path) -> Tuple[bool, str]:
    if not xml_path.exists():
        return False, f"XML file not found: {xml_path}"
    text = xml_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return False, "XML file is empty"
    if not text.endswith("</featureModel>"):
        return False, "XML does not end with </featureModel>"
    try:
        root = ET.fromstring(text)
    except Exception as e:
        return False, f"XML parse error: {e}"
    if root.tag != "featureModel":
        return False, f"Root tag is '{root.tag}', expected 'featureModel'"
    return True, ""


def _configure_llm_for_ollama(model: str, api_key_file: str) -> None:
    if model.strip():
        os.environ["OLLAMA_LLM_MODEL"] = model.strip()
    if api_key_file.strip():
        key_path = Path(api_key_file).expanduser().resolve()
        key = load_key_file(key_path)
        if key:
            os.environ["OLLAMA_API_KEY_FILE"] = str(key_path)


def _build_proprietary_llm(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key_env: str,
    api_key_file: Optional[Path],
    fallback_api_key_file: Optional[Path],
    temperature: float,
    max_tokens: int,
):
    key = os.getenv(api_key_env, "").strip()
    if not key:
        candidate = api_key_file or fallback_api_key_file
        if candidate is None or not candidate.exists():
            raise MissingKeyError(api_key_env, str(candidate) if candidate else f"env:{api_key_env}")
        key = load_key_file(candidate)
        if not key:
            raise MissingKeyError(api_key_env, str(candidate))
        os.environ[api_key_env] = key

    return create_judge_client(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=FIXED_PROVIDER_TIMEOUT_S,
    )


def _interactive_setup(args: argparse.Namespace, cfg_yaml):
    llm_client = None
    high_level_features = None

    mode = prompt_choice("1) Open Source LLM  OR Proprietary LLM", ("Open Source LLM", "Proprietary LLM"))

    if mode == "Open Source LLM":
        args.llm_provider = "ollama"
        args.llm_model = prompt_choice(
            "Select Open Source LLM model",
            ("gpt-oss:120b-cloud", "glm-4.7:cloud", "deepseek-v3.2:cloud"),
        )
        key_path = Path("api_keys/ollama_key.txt")
        key = load_key_file(key_path)
        if key:
            os.environ["OLLAMA_API_KEY_FILE"] = str(key_path)
            os.environ.setdefault("OLLAMA_LLM_HOST", "https://ollama.com")
        else:
            print("WARN: ollama_key not found. Using local Ollama for LLM.")
            os.environ.setdefault("OLLAMA_LLM_HOST", "http://127.0.0.1:11434")
        os.environ.setdefault("OLLAMA_EMBED_HOST", "http://127.0.0.1:11434")
    else:
        args.llm_model = prompt_choice(
            "Select Proprietary LLM model",
            ("gpt-4.1", "claude-opus-4-5", "gemini-3-pro-preview"),
        )
        provider_map = {
            "gpt-4.1": ("openai", "OPENAI_API_KEY"),
            "claude-opus-4-5": ("anthropic", "ANTHROPIC_API_KEY"),
            "gemini-3-pro-preview": ("gemini", "GEMINI_API_KEY"),
        }
        provider, env_var = provider_map[args.llm_model]
        args.llm_provider = provider
        args.api_key_env = env_var

        key_file = cfg_yaml.llm_judge.api_key_dir / f"{provider}_key.txt"
        llm_client = _build_proprietary_llm(
            provider=provider,
            model=args.llm_model,
            base_url=args.llm_base_url or cfg_yaml.llm_judge.base_url,
            api_key_env=env_var,
            api_key_file=None,
            fallback_api_key_file=key_file,
            temperature=args.temperature,
            max_tokens=args.judge_max_tokens,
        )

    args.domain = input("Enter domain [Model Driven Engineering]: ").strip() or "Model Driven Engineering"
    args.root_feature = input("Enter root feature [Model Federation]: ").strip() or "Model Federation"

    args.collection_mode = prompt_choice("Collection mode", ("per_source", "one_collection"))

    default_k_list = DEFAULT_K_VALUES if args.collection_mode == "per_source" else DEFAULT_K_VALUES_ONE_COLLECTION
    default_k_prompt = ",".join(str(k) for k in default_k_list)

    k_mode = prompt_choice("K-value selection", (f"Preset ({default_k_prompt})", "Custom"))
    if k_mode.startswith("Preset"):
        args.k_values = [str(k) for k in default_k_list]
    else:
        while True:
            raw = input(f"Enter k-values (space/comma separated) [{default_k_prompt}]: ").strip() or default_k_prompt
            try:
                args.k_values = [str(k) for k in _parse_k_values([raw])]
                break
            except argparse.ArgumentTypeError as e:
                print(f"Invalid k-values: {e}")
    args.repeats = _prompt_int("Repeats per k", default=max(1, args.repeats), min_value=1)
    args.temperature = _prompt_float("Temperature", default=args.temperature)
    args.max_total_results = _prompt_int("Max total results (0 = no fixed cap)", default=args.max_total_results, min_value=0)
    args.max_retries_per_k = _prompt_int("Max retries per k run", default=max(1, args.max_retries_per_k), min_value=1)

    hl = input("Include high-level features? (Y/n): ").strip().lower()
    feats = default_high_level_features()
    if hl not in ("n", "no"):
        print("\nHigh-level features (default):")
        for k, v in feats.items():
            print(f"- {k}: {v}")
        confirm = input("Use these? (Y/n): ").strip().lower()
        if confirm not in ("n", "no"):
            high_level_features = feats

    return llm_client, high_level_features


def _noninteractive_setup(args: argparse.Namespace, cfg_yaml):
    llm_client = None
    high_level_features = default_high_level_features() if args.use_default_high_level_features else None

    provider = args.llm_provider
    if provider == "ollama":
        _configure_llm_for_ollama(args.llm_model, args.api_key_file)
        return llm_client, high_level_features

    model = args.llm_model or DEFAULT_MODEL_BY_PROVIDER[provider]
    key_env = args.api_key_env or DEFAULT_KEY_ENV_BY_PROVIDER[provider]
    direct_key_file = _resolve_optional_path(args.api_key_file)
    fallback_key_file = cfg_yaml.llm_judge.api_key_dir / f"{provider}_key.txt"

    llm_client = _build_proprietary_llm(
        provider=provider,
        model=model,
        base_url=args.llm_base_url or cfg_yaml.llm_judge.base_url,
        api_key_env=key_env,
        api_key_file=direct_key_file,
        fallback_api_key_file=fallback_key_file,
        temperature=args.temperature,
        max_tokens=args.judge_max_tokens,
    )
    return llm_client, high_level_features


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run SS-RAG k-ablation (generation only): produce FM XML outputs for each k sequentially."
    )
    ap.add_argument("--interactive", action="store_true", help="Run with guided prompts.")
    ap.add_argument("--root-feature", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument(
        "--k-values",
        nargs="+",
        default=[str(k) for k in DEFAULT_K_VALUES],
        help="K values to sweep. Supports space/comma separated values. Default: 20 40 60 80 100",
    )
    ap.add_argument("--repeats", type=int, default=1, help="Runs per k-value (default: 1)")

    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--prompt-path", default="", help="Custom prompt template path")
    ap.add_argument("--max-total-results", type=int, default=150, help="0 => no fixed cap (default: 150)")
    ap.add_argument("--max-total-chars", type=int, default=18_000)
    ap.add_argument("--max-chunk-chars", type=int, default=2_500)
    ap.add_argument("--collection-mode", default="per_source", choices=["per_source", "one_collection"])
    ap.add_argument("--one-collection-name", default="fame_all")
    ap.add_argument("--collection-prefix", default="")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--xsd-path", default="", help="Override XSD path")
    ap.add_argument("--feature-metamodel-path", default="", help="Override feature metamodel path")
    ap.add_argument("--use-default-high-level-features", action="store_true")

    ap.add_argument(
        "--skip-vectorize",
        action="store_true",
        help="Skip vectorization in every run (requires pre-indexed collections).",
    )
    ap.add_argument(
        "--vectorize-each-run",
        action="store_true",
        help="Force vectorization before every run.",
    )

    ap.add_argument("--llm-provider", choices=["ollama", "openai", "anthropic", "gemini"], default="ollama")
    ap.add_argument("--llm-model", default="")
    ap.add_argument("--llm-base-url", default="")
    ap.add_argument("--api-key-env", default="")
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--judge-max-tokens", type=int, default=16000)

    ap.add_argument(
        "--allow-incomplete-xml",
        action="store_true",
        help="If set, do not stop/retry when XML is incomplete or invalid.",
    )
    ap.add_argument("--max-retries-per-k", type=int, default=2, help="Retries for each (k, repeat) if XML is incomplete")

    ap.add_argument("--out-dir", default="results/rag/ss-rgfm/ablation")
    ap.add_argument("--out-manifest", default="", help="Optional explicit manifest output JSON path")

    return ap.parse_args()


def main() -> None:
    cfg_yaml = load_config()
    args = parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.max_retries_per_k < 1:
        raise ValueError("--max-retries-per-k must be >= 1")

    llm_client = None
    high_level_features = None
    if args.interactive:
        llm_client, high_level_features = _interactive_setup(args, cfg_yaml)
    else:
        if not (args.root_feature and args.domain):
            raise ValueError("Provide --root-feature and --domain, or use --interactive.")
        llm_client, high_level_features = _noninteractive_setup(args, cfg_yaml)

    k_values = _parse_k_values(args.k_values)
    chunks_dir = _resolve_optional_path(args.chunks_dir)
    prompt_path = _resolve_optional_path(args.prompt_path) or cfg_yaml.pipelines.ss_rgfm_prompt_path
    xsd_path = _resolve_optional_path(args.xsd_path)
    feature_metamodel_path = _resolve_optional_path(args.feature_metamodel_path)

    # Ensure Ollama model env is set when using open-source provider (avoids fallback to default)
    if args.llm_provider == "ollama" and args.llm_model:
        os.environ["OLLAMA_LLM_MODEL"] = args.llm_model

    paths = build_paths()
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    manifest_path = _resolve_optional_path(args.out_manifest) if args.out_manifest else (out_dir / f"ss_rgfm_k_runs_{ts}.json")
    assert manifest_path is not None
    ensure_dir(manifest_path.parent)

    print("\n==================== SS-RAG K ABLATION ====================")
    print(f"Root feature          : {args.root_feature}")
    print(f"Domain                : {args.domain}")
    print(f"K values              : {k_values}")
    print(f"Repeats per k         : {args.repeats}")
    print(f"Model                 : {getattr(llm_client, 'model', None) or os.getenv('OLLAMA_LLM_MODEL', 'ollama-default')}")
    print(f"Require complete XML  : {not args.allow_incomplete_xml}")
    print(f"Max retries per run   : {args.max_retries_per_k}")
    print("============================================================")

    runs: List[Dict[str, object]] = []
    global_run_index = 0

    for k in k_values:
        print(f"\n---- Starting k={k} ----")

        for rep in range(1, args.repeats + 1):
            completed = False
            last_err = ""

            for attempt in range(1, args.max_retries_per_k + 1):
                global_run_index += 1

                if args.skip_vectorize:
                    skip_vectorize = True
                elif args.vectorize_each_run:
                    skip_vectorize = False
                else:
                    skip_vectorize = global_run_index > 1

                max_total_results = args.max_total_results if args.max_total_results > 0 else 1_000_000
                run_tag = f"ss-rgfm-k{k}-r{rep}" + (f"-a{attempt}" if attempt > 1 else "")

                cfg = SSRGFMConfig(
                    root_feature=args.root_feature,
                    domain=args.domain,
                    chunks_dir=chunks_dir,
                    chunks_files=None,
                    collection_mode=args.collection_mode,
                    one_collection_name=args.one_collection_name,
                    collection_prefix=args.collection_prefix,
                    batch_size=args.batch_size,
                    n_results_per_collection=k,
                    max_total_results=max_total_results,
                    max_total_chars=args.max_total_chars,
                    max_chunk_chars=args.max_chunk_chars,
                    k_strategy="fixed",
                    prompt_path=prompt_path,
                    temperature=args.temperature,
                    xsd_path=xsd_path,
                    feature_metamodel_path=feature_metamodel_path,
                    high_level_features=high_level_features,
                    run_tag=run_tag,
                )

                print(f"Running k={k}, repeat={rep}, attempt={attempt} (skip_vectorize={skip_vectorize})")
                out = run_ss_rgfm(cfg, llm=llm_client, skip_vectorize=skip_vectorize)

                fm_xml = Path(out["fm_xml"]).expanduser().resolve()
                meta_file = Path(out["meta"]).expanduser().resolve()
                meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

                xml_ok, xml_error = _check_complete_feature_model_xml(fm_xml)
                run_record = {
                    "k_value": k,
                    "repeat": rep,
                    "attempt": attempt,
                    "run_id": out.get("run_id"),
                    "xml_ok": xml_ok,
                    "xml_error": xml_error if not xml_ok else "",
                    "fm_xml": str(fm_xml),
                    "prompt": out.get("prompt"),
                    "evidence": out.get("evidence"),
                    "collection_mode": args.collection_mode,
                    "meta": str(meta_file),
                    "num_evidence_chunks": meta.get("num_evidence_chunks"),
                    "n_results_per_collection_effective": meta.get("n_results_per_collection_effective"),
                    "llm_model": meta.get("llm_model", getattr(llm_client, "model", os.getenv("OLLAMA_LLM_MODEL", "unknown"))),
                    "llm_duration_seconds": meta.get("llm_duration_seconds"),
                    "prompt_saved": meta.get("prompt_saved"),
                }
                runs.append(run_record)

                if args.allow_incomplete_xml or xml_ok:
                    completed = True
                    print(f"Completed k={k}, repeat={rep}, attempt={attempt}: {fm_xml}")
                    break

                last_err = xml_error
                print(f"WARN: Incomplete/invalid XML for k={k}, repeat={rep}, attempt={attempt}: {xml_error}")

            if not completed:
                raise RuntimeError(
                    f"Failed k={k}, repeat={rep} after {args.max_retries_per_k} attempts. "
                    f"Last error: {last_err}"
                )

        print(f"---- Finished k={k}. All repeats completed ----")

    payload = {
        "created_at": ts,
        "root_feature": args.root_feature,
        "domain": args.domain,
        "k_values": k_values,
        "repeats": args.repeats,
        "llm_provider": args.llm_provider,
        "llm_model": getattr(llm_client, "model", None) or os.getenv("OLLAMA_LLM_MODEL", ""),
        "temperature": args.temperature,
        "max_total_results": args.max_total_results,
        "max_total_chars": args.max_total_chars,
        "max_chunk_chars": args.max_chunk_chars,
        "collection_mode": args.collection_mode,
        "chunks_dir": str(chunks_dir) if chunks_dir else str(paths.processed_data / "chunks"),
        "prompt_path": str(prompt_path) if prompt_path else None,
        "allow_incomplete_xml": args.allow_incomplete_xml,
        "max_retries_per_k": args.max_retries_per_k,
        "runs": runs,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSUCCESS: K-ablation generation finished")
    print(f"Manifest: {manifest_path}")
    print(f"Generated runs: {len(runs)}")


if __name__ == "__main__":
    logger = get_logger("ss_rgfm_k_ablation")
    try:
        main()
    except UserMessageError as e:
        print(f"ERROR: {format_error(e)} (see results/logs/fame.log for details)")
        log_exception(logger, e)
    except Exception as e:
        log_exception(logger, e)
        raise
