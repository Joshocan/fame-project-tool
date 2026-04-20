from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    _HAS_ST = False

from .feature_list import extract_feature_list


@dataclass(frozen=True)
class ProxyEvidenceConfig:
    model_name: str = 'all-mpnet-base-v2'
    trace_presence_weight: float = 0.2
    trace_support_weight: float = 0.4
    label_support_weight: float = 0.4
    support_threshold: float = 0.35
    max_chunks: int = 0
    min_text_chars: int = 10
    prefer_evidence_refs: bool = True


@dataclass(frozen=True)
class ProxyEvidenceResult:
    num_concrete_features: int
    trace_present_rate: float
    label_support_mean: float
    trace_support_mean: float
    supported_feature_rate: float
    evidence_score: float


@dataclass(frozen=True)
class ProxyEvidenceContext:
    model: Any
    chunks: List[Dict[str, Any]]
    chunk_embeddings: Any


_TRACE_BLOCK_RE = re.compile(r'Trace:\s*\[(.*?)\]', re.IGNORECASE | re.DOTALL)
_NON_WORD_RE = re.compile(r'[^a-z0-9]+')

_MODEL_CACHE: dict[str, Any] = {}


def _normalize_token(text: str) -> str:
    return _NON_WORD_RE.sub('', (text or '').lower())


def _strip_trace_block(text: str) -> str:
    cleaned = _TRACE_BLOCK_RE.sub('', text or '')
    return ' '.join(cleaned.split()).strip()


def _maybe_load_hf_token() -> None:
    if os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN'):
        return
    repo_root = Path.cwd()
    token_path = repo_root / 'api_keys' / 'hf_key.txt'
    if not token_path.exists():
        return
    token = token_path.read_text(encoding='utf-8', errors='ignore').strip()
    if token:
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGINGFACE_HUB_TOKEN'] = token


def _get_model(model_name: str) -> Any:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    _maybe_load_hf_token()
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


def _load_chunks(chunks_dir: Path, *, max_chunks: int = 0, min_text_chars: int = 10) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for path in sorted(chunks_dir.glob('*.chunks.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            continue
        for row in payload.get('chunks', []):
            text = str(row.get('text') or '').strip()
            if len(text) < min_text_chars:
                continue
            chunks.append(
                {
                    'chunk_id': str(row.get('chunk_id') or ''),
                    'source': str(row.get('source') or payload.get('source') or path.name),
                    'text': text,
                    'metadata': row.get('metadata') or {},
                }
            )
            if max_chunks > 0 and len(chunks) >= max_chunks:
                return chunks
    return chunks


def _candidate_indices_for_refs(chunks: Sequence[Dict[str, Any]], refs: Sequence[str]) -> List[int]:
    if not refs:
        return list(range(len(chunks)))
    ref_tokens = {_normalize_token(r) for r in refs if _normalize_token(r)}
    if not ref_tokens:
        return list(range(len(chunks)))

    indices: List[int] = []
    for idx, chunk in enumerate(chunks):
        source = _normalize_token(Path(str(chunk.get('source') or '')).stem)
        chunk_id = _normalize_token(str(chunk.get('chunk_id') or ''))
        if any(tok and (tok in source or source in tok or tok in chunk_id) for tok in ref_tokens):
            indices.append(idx)
    return indices or list(range(len(chunks)))


def _best_similarity(query: str, chunk_embeddings, model: Any, indices: Sequence[int]) -> float:
    query = ' '.join((query or '').split()).strip()
    if not query:
        return 0.0
    q_emb = model.encode([query], normalize_embeddings=True, convert_to_tensor=True)
    sims = util.cos_sim(q_emb[0], chunk_embeddings)[0]
    if indices:
        vals = [float(sims[i]) for i in indices]
        return max(vals) if vals else 0.0
    return float(np.max(sims.cpu().numpy())) if hasattr(sims, 'cpu') else float(np.max(sims))


def prepare_evidence_context(
    chunks_dir: Path | str,
    cfg: ProxyEvidenceConfig,
    *,
    model: Any | None = None,
) -> ProxyEvidenceContext:
    if not _HAS_ST:
        raise ImportError('sentence_transformers is required for proxy evidence scoring but is not installed.')

    chunks_dir = Path(chunks_dir)
    if not chunks_dir.exists():
        raise FileNotFoundError(f'Chunks directory not found: {chunks_dir}')

    chunks = _load_chunks(chunks_dir, max_chunks=max(0, int(cfg.max_chunks)), min_text_chars=max(1, int(cfg.min_text_chars)))
    st_model = model or _get_model(cfg.model_name)
    if chunks:
        chunk_texts = [str(c['text']) for c in chunks]
        chunk_embeddings = st_model.encode(chunk_texts, normalize_embeddings=True, convert_to_tensor=True)
    else:
        chunk_embeddings = None
    return ProxyEvidenceContext(model=st_model, chunks=chunks, chunk_embeddings=chunk_embeddings)


def score_evidence(
    xml_path: Path | str,
    chunks_dir: Path | str,
    cfg: ProxyEvidenceConfig,
    *,
    model: Any | None = None,
    context: ProxyEvidenceContext | None = None,
) -> ProxyEvidenceResult:
    if not _HAS_ST:
        raise ImportError('sentence_transformers is required for proxy evidence scoring but is not installed.')

    xml_path = Path(xml_path)
    records = extract_feature_list(xml_path)
    concrete = [rec for rec in records if rec.feature_type == 'concrete']
    if not concrete:
        return ProxyEvidenceResult(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    evidence_ctx = context or prepare_evidence_context(chunks_dir, cfg, model=model)
    chunks = evidence_ctx.chunks
    if not chunks or evidence_ctx.chunk_embeddings is None:
        return ProxyEvidenceResult(len(concrete), 0.0, 0.0, 0.0, 0.0, 0.0)

    st_model = evidence_ctx.model
    chunk_embeddings = evidence_ctx.chunk_embeddings

    trace_present_count = 0
    label_scores: List[float] = []
    trace_scores: List[float] = []
    supported = 0

    for rec in concrete:
        indices = _candidate_indices_for_refs(chunks, rec.evidence_refs) if cfg.prefer_evidence_refs else list(range(len(chunks)))
        label_score = _best_similarity(rec.feature_name, chunk_embeddings, st_model, indices)
        trace_text = _strip_trace_block(rec.description)
        trace_score = _best_similarity(trace_text, chunk_embeddings, st_model, indices) if trace_text else 0.0
        if trace_text:
            trace_present_count += 1
        if max(label_score, trace_score) >= float(cfg.support_threshold):
            supported += 1
        label_scores.append(label_score)
        trace_scores.append(trace_score)

    trace_present_rate = trace_present_count / len(concrete)
    label_support_mean = float(np.mean(label_scores)) if label_scores else 0.0
    trace_support_mean = float(np.mean(trace_scores)) if trace_scores else 0.0
    supported_feature_rate = supported / len(concrete)

    evidence_score = (
        float(cfg.trace_presence_weight) * trace_present_rate
        + float(cfg.trace_support_weight) * trace_support_mean
        + float(cfg.label_support_weight) * label_support_mean
    )
    evidence_score = max(0.0, min(1.0, evidence_score))

    return ProxyEvidenceResult(
        num_concrete_features=len(concrete),
        trace_present_rate=round(trace_present_rate, 6),
        label_support_mean=round(label_support_mean, 6),
        trace_support_mean=round(trace_support_mean, 6),
        supported_feature_rate=round(supported_feature_rate, 6),
        evidence_score=round(evidence_score, 6),
    )
