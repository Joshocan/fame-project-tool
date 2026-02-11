from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fame.utils.dirs import build_paths, ensure_dir, ensure_for_stage
from fame.vectorization.pipeline import default_collection_name
from fame.retrieval.service import RetrievalService
from fame.nonrag.prompting import render_prompt_template, serialize_high_level_features
from fame.nonrag.llm_ollama_http import OllamaHTTP, assert_ollama_running
from fame.utils.placeholder_check import assert_no_placeholders, UnresolvedPlaceholdersError
from fame.exceptions import PlaceholderError, MissingChunksError
from fame.evaluation import start_timer, elapsed_seconds


@dataclass(frozen=True)
class ISRgfmConfig:
    root_feature: str
    domain: str

    chunks_dir: Optional[Path] = None
    chunks_files: Optional[Sequence[Path]] = None

    initial_prompt_path: Optional[Path] = None
    iter_prompt_path: Optional[Path] = None

    n_results_per_collection: Optional[int] = None
    max_total_results: Optional[int] = None
    max_total_chars: int = 18_000
    max_chunk_chars: int = 2_500

    xsd_path: Optional[Path] = None
    feature_metamodel_path: Optional[Path] = None
    high_level_features: Optional[Dict[str, str]] = None
    temperature: float = 0.2

    run_tag: str = "is-rgfm"


def _default_chunks_dir(paths) -> Path:
    return paths.processed_data / "chunks"


def _list_chunks_files(chunks_dir: Path) -> List[Path]:
    return sorted(chunks_dir.glob("*.chunks.json"))


def run_is_rgfm(cfg: ISRgfmConfig, *, llm: Optional[object] = None, retriever: Optional[RetrievalService] = None) -> Dict[str, str]:
    paths = build_paths()
    ensure_for_stage("is-rgfm", paths)
    ensure_for_stage("preprocess", paths)

    # LLM
    if llm is None:
        assert_ollama_running()
        llm = OllamaHTTP()
    model_name = getattr(llm, "model", "unknown")

    # Files
    chunks_dir = cfg.chunks_dir or _default_chunks_dir(paths)
    ensure_dir(chunks_dir)
    files = list(cfg.chunks_files) if cfg.chunks_files else _list_chunks_files(chunks_dir)
    if not files:
        raise MissingChunksError(str(chunks_dir))

    retr = retriever or RetrievalService()

    spec_dir = paths.specifications
    xsd_path = cfg.xsd_path or (spec_dir / "feature_model_featureide.xsd")
    metamodel_path = cfg.feature_metamodel_path or (spec_dir / "feature_metamodel_specification.txt")
    xsd_text = xsd_path.read_text(encoding="utf-8") if xsd_path.exists() else ""
    metamodel_text = metamodel_path.read_text(encoding="utf-8") if metamodel_path.exists() else ""
    high_level_xml = serialize_high_level_features(cfg.high_level_features)

    previous_xml = ""
    iteration_meta = []
    ts_start = time.strftime("%Y-%m-%dT%H-%M-%S")
    run_id_base = f"is_rgfm_response_{re.sub(r'[^a-zA-Z0-9]+', '-', model_name).strip('-').lower() or 'unknown'}_{ts_start}"

    for i, f in enumerate(files, start=1):
        collection = default_collection_name(f)
        res = retr.retrieve(
            root_feature=cfg.root_feature,
            domain=cfg.domain,
            collections=[collection],
            n_results_per_collection=cfg.n_results_per_collection or 6,
            max_total_results=cfg.max_total_results or 12,
        )
        context = retr.to_prompt_evidence(
            res,
            max_total_chars=cfg.max_total_chars,
            max_chunk_chars=cfg.max_chunk_chars,
        )

        tmpl_text = (
            Path(cfg.initial_prompt_path).read_text(encoding="utf-8")
            if cfg.initial_prompt_path and Path(cfg.initial_prompt_path).exists() and i == 1
            else Path(cfg.iter_prompt_path).read_text(encoding="utf-8")
            if cfg.iter_prompt_path and Path(cfg.iter_prompt_path).exists() and i > 1
            else ""
        )
        if not tmpl_text:
            # fallback to defaults
            default_init = (paths.base_dir / "prompts" / "fm_extraction_prompt.txt")
            default_iter = (paths.base_dir / "prompts" / "fm_iterated_prompt.txt")
            tmpl_path = default_init if i == 1 else default_iter
            tmpl_text = tmpl_path.read_text(encoding="utf-8")

        values = {
            "ROOT_FEATURE": cfg.root_feature,
            "DOMAIN": cfg.domain,
            "CONTEXT": context,
            "PREVIOUS_FM_XML": previous_xml or "(empty)",
            "HIGH_LEVEL_FEATURES": high_level_xml,
            "XSD_METAMODEL": xsd_text,
            "FEATURE_METAMODEL": metamodel_text,
        }

        prompt = render_prompt_template(tmpl_text, values=values, strict=True)
        try:
            assert_no_placeholders(prompt)
        except UnresolvedPlaceholdersError as e:
            raise PlaceholderError(e.placeholders) from e

        print(f"⏳ Iteration {i}/{len(files)} — source: {f.name}")
        t0 = start_timer()
        out_xml = llm.generate(prompt, temperature=cfg.temperature)
        iter_duration = elapsed_seconds(t0)
        previous_xml = out_xml

        iter_tag = f"{run_id_base}_iter{i:02d}_{f.stem}"
        prompt_file = paths.reports / f"{iter_tag}.prompt.txt"
        context_file = paths.results / "rag" / "is-rgfm" / "context" / f"{iter_tag}.context.txt"
        xml_file = paths.results / "rag" / "is-rgfm" / "runs" / f"{iter_tag}.xml"
        ensure_dir(prompt_file.parent)
        ensure_dir(context_file.parent)
        ensure_dir(xml_file.parent)

        prompt_file.write_text(prompt, encoding="utf-8")
        context_file.write_text(context, encoding="utf-8")
        xml_file.write_text(out_xml, encoding="utf-8")

        iteration_meta.append(
            {
                "iter": i,
                "source_chunks": str(f),
                "collection": collection,
                "prompt": str(prompt_file),
                "context": str(context_file),
                "xml": str(xml_file),
                "llm_duration_seconds": iter_duration,
            }
        )

    final_xml_file = paths.is_fm / f"{run_id_base}.final.xml"
    ensure_dir(final_xml_file.parent)
    final_xml_file.write_text(previous_xml, encoding="utf-8")

    meta_file = paths.is_reports / f"{run_id_base}.meta.json"
    meta = {
        "run_id": run_id_base,
        "root_feature": cfg.root_feature,
        "domain": cfg.domain,
        "num_sources": len(files),
        "llm_host": getattr(llm, "host", getattr(llm, "base_url", "")),
        "llm_model": model_name,
        "total_llm_duration_seconds": sum(float(m.get("llm_duration_seconds", 0)) for m in iteration_meta),
        "iterations": iteration_meta,
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"SUCCESS: IS-RGFM finished. Final FM: {final_xml_file}")

    return {
        "final_xml": str(final_xml_file),
        "meta": str(meta_file),
        "run_id": run_id_base,
    }
