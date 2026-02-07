# fame/nonrag/is_pipeline.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Protocol

from fame.context import ContextBuildConfig, ContextManager, chunks_from_chunks_json
from fame.ingestion.pipeline import ingest_and_prepare
from fame.utils.dirs import build_paths, ensure_for_stage, ensure_dir
from fame.nonrag.prompting import render_prompt_template, serialize_high_level_features
from fame.evaluation import start_timer, elapsed_seconds
from fame.utils.placeholder_check import assert_no_placeholders, UnresolvedPlaceholdersError
from fame.exceptions import PlaceholderError, MissingChunksError

from .llm_ollama_http import OllamaHTTP, assert_ollama_running


class LLM(Protocol):
    def generate(self, prompt: str, *, system: Optional[str] = None, temperature: float = 0.2) -> str: ...


DEFAULT_IS_NONRAG_PROMPT = """You are an expert Software Product Line (SPL) Engineer.

TASK:
Incrementally construct (and refine) a Feature Model for:
ROOT FEATURE: {root_feature}
DOMAIN: {domain}

RULES:
- Use ONLY the evidence provided below.
- Preserve previously established structure unless contradicted by new evidence.
- If you change something, do so explicitly and consistently.

PREVIOUS OUTPUT (may be empty on first iteration):
{previous_output}

NEW EVIDENCE (delta):
{delta_context}

OUTPUT FORMAT:
Return ONLY the updated XML feature model (no markdown, no explanation).
"""

# default iterative prompt (if file not found)
DEFAULT_IS_NONRAG_ITER_PROMPT = """You are an expert SPL Engineer.

TASK:
Merge/extend the existing Feature Model with NEW evidence.

ROOT FEATURE: {root_feature}
DOMAIN: {domain}

PREVIOUS FM (XML):
{previous_output}

NEW EVIDENCE (delta):
{delta_context}

OUTPUT:
Return ONLY the updated XML feature model (no markdown).
"""


@dataclass(frozen=True)
class ISNonRagConfig:
    root_feature: str
    domain: str

    chunks_dir: Optional[Path] = None
    chunks_files: Optional[Sequence[Path]] = None

    # context budgets
    max_delta_chars: int = 50_000
    max_delta_chunks: int = 50
    max_delta_chunk_chars: int = 6_000

    # output naming
    run_tag: str = "is-nonrag"

    # prompt overrides
    initial_prompt_path: Optional[Path] = None  # used only for first iteration
    iter_prompt_path: Optional[Path] = None     # used for subsequent iterations

    # high-level features (optional)
    high_level_features: Optional[Dict[str, str]] = None
    max_depth: Optional[int] = None

    # generation params
    temperature: float = 0.2


def _default_chunks_dir(paths) -> Path:
    return paths.processed_data / "chunks"


def _list_chunks_files(chunks_dir: Path) -> List[Path]:
    return sorted(chunks_dir.glob("*.chunks.json"))


def _load_template(path: Optional[Path], default_text: str) -> str:
    if path:
        p = Path(path).expanduser().resolve()
        if p.exists():
            return p.read_text(encoding="utf-8")
    return default_text


def run_is_nonrag(cfg: ISNonRagConfig, llm: Optional[LLM] = None) -> Dict[str, str]:
    """
    Iterated-stage Non-RAG:
      - iterate over chunks.json files (article-by-article)
      - append ONLY new evidence (delta context) per iteration
      - call LLM each iteration to update FM
      - save per-iteration prompt/output and final FM

    Returns dict with final artifact paths.
    """
    print(
        """
===========================================================
IS-NONRAG PIPELINE — Iterative Non-RAG Feature Modeling
===========================================================
For each source file, a delta context is added and the FM is
refined. No retrieval or vector DB is used.
"""
    )

    paths = build_paths()
    ensure_for_stage("is-nonrag", paths)
    ensure_for_stage("preprocess", paths)
    spec_dir = paths.specifications
    xsd_path = spec_dir / "feature_model_schema.xsd"
    metamodel_path = spec_dir / "feature_metamodel_specification.txt"
    xsd_text = xsd_path.read_text(encoding="utf-8") if xsd_path.exists() else ""
    metamodel_text = metamodel_path.read_text(encoding="utf-8") if metamodel_path.exists() else ""

    # LLM
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

    cm = ContextManager()
    delta_cfg = ContextBuildConfig(
        max_total_chars=cfg.max_delta_chars,
        max_chunks=cfg.max_delta_chunks,
        max_chunk_chars=cfg.max_delta_chunk_chars,
        include_headers=True,
        order="by_page_then_id",
    )

    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    model_name = ""
    run_id_base = f"{cfg.run_tag}_{ts}"

    previous_xml = ""  # first iteration has none
    iteration_meta: List[Dict[str, str]] = []

    print(f"🧩 Sources to process: {len(files)}")

    for i, f in enumerate(files, start=1):
        chunks = chunks_from_chunks_json(f)

        # First file: treat as "initial context" but still stored as a delta block for consistency
        title = f"IN-LINE CONTEXT ADDITION (iter={i}, source={f.name})"
        if i == 1:
            delta_context = cm.add_initial_context(chunks, delta_cfg, title=title)
        else:
            delta_context = cm.add_delta_context(chunks, delta_cfg, title=title)

        print(f"⏳ Iteration {i}/{len(files)} — source: {f.name}")
        # Choose template: initial vs iterative (from config or defaults)
        if i == 1:
            tmpl_text = _load_template(cfg.initial_prompt_path, DEFAULT_IS_NONRAG_PROMPT)
        else:
            tmpl_text = _load_template(cfg.iter_prompt_path, DEFAULT_IS_NONRAG_ITER_PROMPT)

        high_level_xml = serialize_high_level_features(cfg.high_level_features)
        values = {
            "root_feature": cfg.root_feature,
            "ROOT_FEATURE": cfg.root_feature,
            "domain": cfg.domain,
            "DOMAIN": cfg.domain,
            "previous_output": (previous_xml.strip() or "(empty)"),
            "PREVIOUS_FM_XML": (previous_xml.strip() or "(empty)"),
            "delta_context": delta_context,
            "DELTA_CONTEXT": delta_context,
            "HIGH_LEVEL_FEATURES": high_level_xml,
            "MAX_DEPTH": str(cfg.max_depth) if cfg.max_depth is not None else "",
            "XSD_METAMODEL": xsd_text,
            "FEATURE_METAMODEL": metamodel_text,
            "INPUT_TEXT": delta_context,
            "inputfile": "",
            "mmdir": "",
        }

        prompt = render_prompt_template(tmpl_text, values=values, strict=False)
        try:
            assert_no_placeholders(prompt)
        except UnresolvedPlaceholdersError as e:
            raise PlaceholderError(e.placeholders) from e

        print("   → Calling LLM...")
        t_llm = start_timer()
        out_xml = llm.generate(prompt, temperature=cfg.temperature)
        iter_duration = elapsed_seconds(t_llm)
        previous_xml = out_xml
        model_name = getattr(llm, "model", "") or model_name

        # save per-iteration artifacts
        iter_tag = f"{run_id_base}_iter{i:02d}_{f.stem}"
        prompt_file = paths.non_is_runs / f"{iter_tag}.prompt.txt"
        delta_file = paths.non_is_context / f"{iter_tag}.delta.txt"
        out_file = paths.non_is_runs / f"{iter_tag}.xml"

        prompt_file.write_text(prompt, encoding="utf-8")
        delta_file.write_text(delta_context, encoding="utf-8")
        out_file.write_text(out_xml, encoding="utf-8")

        iteration_meta.append(
            {
                "iter": str(i),
                "source_chunks": str(f),
                "prompt": str(prompt_file),
                "delta": str(delta_file),
                "xml": str(out_file),
                "llm_duration_seconds": iter_duration,
            }
        )

    # final outputs
    model_safe = (model_name or "").replace(" ", "-").replace("/", "-").replace(":", "-")
    run_id = (
        f"is_nonrag_response_{model_safe}_{ts}"
        if model_name
        else f"is_nonrag_response_{ts}"
    )

    final_xml_file = paths.non_is_fm / f"{run_id}.final.xml"
    meta_file = paths.non_is_reports / f"{run_id}.meta.json"

    final_xml_file.write_text(previous_xml, encoding="utf-8")

    meta = {
        "run_id": run_id,
        "root_feature": cfg.root_feature,
        "domain": cfg.domain,
        "num_sources": len(files),
        "ollama_host": getattr(llm, "host", ""),
        "ollama_model": getattr(llm, "model", ""),
        "total_llm_duration_seconds": sum(float(m.get("llm_duration_seconds", 0)) for m in iteration_meta),
        "iterations": iteration_meta,
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ IS-NonRAG finished. Final FM: {final_xml_file}")

    return {
        "final_xml": str(final_xml_file),
        "meta": str(meta_file),
        "run_id": run_id,
    }
