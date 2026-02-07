from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fame.utils.dirs import build_paths, ensure_for_stage, ensure_dir
from fame.ingestion.pipeline import ingest_and_prepare
from fame.vectorization.pipeline import vectorize_from_chunks_dir
from fame.retrieval.service import RetrievalService
from fame.nonrag.llm_ollama_http import OllamaHTTP, assert_ollama_running
from fame.nonrag.prompt_utils import save_modified_prompt
from fame.utils.placeholder_check import assert_no_placeholders
from fame.evaluation import start_timer, elapsed_seconds
from fame.exceptions import MissingChunksError
from fame.loggers import get_logger


@dataclass(frozen=True)
class SSRGFMConfig:
    root_feature: str
    domain: str

    # ingestion/chunks
    chunks_dir: Optional[Path] = None
    chunks_files: Optional[Sequence[Path]] = None

    # vectorization
    collection_mode: str = "per_source"   # per_source | one_collection
    one_collection_name: str = "fame_all"
    collection_prefix: str = ""
    batch_size: int = 24

    # retrieval
    n_results_per_collection: int = 6
    max_total_results: int = 12
    max_total_chars: int = 18_000
    max_chunk_chars: int = 2_500

    # prompt + generation
    prompt_path: Optional[Path] = None
    temperature: float = 0.2

    # output naming
    run_tag: str = "ss-rgfm"


def _default_chunks_dir(paths) -> Path:
    return paths.processed_data / "chunks"


def _list_chunks_files(chunks_dir: Path) -> List[Path]:
    return sorted(chunks_dir.glob("*.chunks.json"))


def _collection_name_for_file(fp: Path, prefix: str = "") -> str:
    # paper.pdf.chunks.json -> paper_pdf (safe-ish)
    stem = fp.stem.replace(".chunks", "").replace(".", "_").replace("-", "_")
    return f"{prefix}{stem}" if prefix else stem


def load_ss_rgfm_prompt(prompt_path: Optional[Path] = None) -> str:
    """
    Load prompt template for SS-RGFM. Falls back to a minimal built-in template.
    """
    if prompt_path:
        p = Path(prompt_path).expanduser()
        if p.exists():
            return p.read_text(encoding="utf-8")
    return (
        "You are an expert in Feature Modeling. Given the domain \"{{DOMAIN}}\" and "
        "root feature \"{{ROOT_FEATURE}}\", produce a FeatureIDE-compatible XML "
        "feature model using the EVIDENCE below.\n\n"
        "EVIDENCE:\n{{EVIDENCE}}\n"
    )


def render_ss_rgfm_prompt(*, root_feature: str, domain: str, evidence: str, prompt_template: str) -> str:
    prompt = (
        prompt_template.replace("{{ROOT_FEATURE}}", root_feature)
        .replace("{{DOMAIN}}", domain)
        .replace("{{EVIDENCE}}", evidence)
    )
    assert_no_placeholders(prompt)
    return prompt


def run_ss_rgfm(
    cfg: SSRGFMConfig,
    *,
    llm: Optional[object] = None,
    retriever: Optional[object] = None,
    skip_vectorize: bool = False,
) -> Dict[str, str]:
    """
    SS-RGFM:
      - ensure Ollama server running (for generation)
      - ensure chunks exist (ingestion if needed)
      - vectorize chunks into Chroma
      - retrieve evidence using default query template
      - build prompt and call LLM once
      - save FM + evidence + prompt + meta
    """
    logger = get_logger("ss_rgfm")
    paths = build_paths()
    ensure_for_stage("ss-rgfm", paths)
    ensure_for_stage("preprocess", paths)
    ensure_for_stage("vectorize", paths)

    # LLM server for generation (Ollama) unless provided
    if llm is None:
        assert_ollama_running()
        llm = OllamaHTTP()

    # Ensure chunks exist
    chunks_dir = cfg.chunks_dir or _default_chunks_dir(paths)
    ensure_dir(chunks_dir)

    files = list(cfg.chunks_files) if cfg.chunks_files else _list_chunks_files(chunks_dir)
    if not files:
        ingest_and_prepare(raw_dir=paths.raw_data, out_dir=chunks_dir)
        files = _list_chunks_files(chunks_dir)

    if not files:
        raise MissingChunksError(str(chunks_dir))

    vec_out = None
    if not skip_vectorize:
        vec_out = vectorize_from_chunks_dir(
            chunks_dir=chunks_dir,
            collection_mode=cfg.collection_mode,
            one_collection_name=cfg.one_collection_name,
            collection_prefix=cfg.collection_prefix,
            batch_size=cfg.batch_size,
        )

    # Determine collections
    if cfg.collection_mode == "one_collection":
        collections = [cfg.one_collection_name]
    else:
        collections = [_collection_name_for_file(f, prefix=cfg.collection_prefix) for f in files]

    retr = retriever or RetrievalService()
    res = retr.retrieve(
        root_feature=cfg.root_feature,
        domain=cfg.domain,
        collections=collections,
        n_results_per_collection=cfg.n_results_per_collection,
        max_total_results=cfg.max_total_results,
    )
    evidence = retr.to_prompt_evidence(
        res,
        max_total_chars=cfg.max_total_chars,
        max_chunk_chars=cfg.max_chunk_chars,
    )

    tmpl = load_ss_rgfm_prompt(cfg.prompt_path)
    prompt = render_ss_rgfm_prompt(
        root_feature=cfg.root_feature,
        domain=cfg.domain,
        evidence=evidence,
        prompt_template=tmpl,
    )

    t0 = start_timer()
    fm_xml = llm.generate(prompt, temperature=cfg.temperature)
    llm_duration = elapsed_seconds(t0)

    # Persist artifacts
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    model_safe = getattr(llm, "model", "unknown-model").replace(":", "-").replace("/", "-")
    prompt_saved = save_modified_prompt(
        prompt=prompt,
        model_safe=model_safe,
        ts=ts,
        paths=paths,
        pipeline_type="ss_rgfm",
    )
    run_id = f"{cfg.run_tag}_response_{model_safe}_{ts}"

    fm_file = paths.ss_fm / f"{run_id}.xml"
    prompt_file = paths.reports / f"{run_id}.prompt.txt"
    evidence_file = paths.reports / f"{run_id}.evidence.txt"
    meta_file = paths.reports / f"{run_id}.meta.json"

    fm_file.write_text(fm_xml, encoding="utf-8")
    prompt_file.write_text(prompt, encoding="utf-8")
    evidence_file.write_text(evidence, encoding="utf-8")

    meta = {
        "run_id": run_id,
        "root_feature": cfg.root_feature,
        "domain": cfg.domain,
        "collection_mode": cfg.collection_mode,
        "collections": collections,
        "vectorization": vec_out,
        "query_used": res.query,
        "num_evidence_chunks": len(res.chunks),
        "ollama_host": llm.host,
        "ollama_model": llm.model,
        "llm_duration_seconds": llm_duration,
        "chunks_dir": str(chunks_dir),
        "prompt_saved": str(prompt_saved),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "ss_rgfm completed",
        extra={
            "run_id": run_id,
            "model": getattr(llm, "model", "unknown"),
            "collections": collections,
            "num_chunks": len(res.chunks),
        },
    )

    return {
        "fm_xml": str(fm_file),
        "prompt": str(prompt_file),
        "evidence": str(evidence_file),
        "meta": str(meta_file),
        "run_id": run_id,
    }
