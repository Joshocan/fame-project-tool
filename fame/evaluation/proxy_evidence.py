from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List
import xml.etree.ElementTree as ET


def normalize_text(text: str) -> str:
    return ' '.join((text or '').replace('_', ' ').replace('-', ' ').lower().split())


def tokenize(text: str) -> set[str]:
    return set(normalize_text(text).split())


def lexical_similarity(a: str, b: str) -> float:
    ta = tokenize(a)
    tb = tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def load_chunk_texts(chunks_dir: str | Path) -> List[str]:
    root = Path(chunks_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f'Chunk directory not found: {root}')
    texts: List[str] = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == '.json':
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                continue
            texts.extend(_extract_texts(data))
        elif suffix in {'.jsonl', '.jl'}:
            for line in path.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    texts.extend(_extract_texts(json.loads(line)))
                except Exception:
                    continue
        elif suffix in {'.txt', '.md'}:
            text = path.read_text(encoding='utf-8').strip()
            if text:
                texts.append(text)
    deduped = []
    seen = set()
    for text in texts:
        norm = normalize_text(text)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(text)
    return deduped


def _extract_texts(data: object) -> List[str]:
    out: List[str] = []
    if isinstance(data, str):
        if data.strip():
            out.append(data)
    elif isinstance(data, dict):
        for key in ('text', 'content', 'chunk', 'page_content', 'body', 'description'):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value)
        for value in data.values():
            out.extend(_extract_texts(value))
    elif isinstance(data, list):
        for item in data:
            out.extend(_extract_texts(item))
    return out


def _extract_feature_records(xml_path: Path) -> List[Dict[str, str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    records: List[Dict[str, str]] = []
    for elem in root.iter():
        name = (elem.attrib.get('name') or elem.attrib.get('feature') or '').strip()
        if not name:
            continue
        desc = (elem.attrib.get('description') or '').strip()
        if not desc:
            for child in elem:
                if child.tag.lower().endswith('description') and (child.text or '').strip():
                    desc = child.text.strip()
                    break
        records.append({'name': name, 'description': desc})
    return records


def _best_support(text: str, chunks: Iterable[str]) -> float:
    scores = [lexical_similarity(text, chunk) for chunk in chunks if chunk]
    return max(scores) if scores else 0.0


def score_evidence(xml_path: str | Path, chunks_dir: str | Path) -> Dict[str, float]:
    xml_path = Path(xml_path).expanduser().resolve()
    records = _extract_feature_records(xml_path)
    chunks = load_chunk_texts(chunks_dir)
    if not records:
        return {
            'trace_present_rate': 0.0,
            'label_support_mean': 0.0,
            'trace_support_mean': 0.0,
            'evidence_score': 0.0,
        }
    trace_present = 0
    label_scores: List[float] = []
    trace_scores: List[float] = []
    for rec in records:
        label_scores.append(_best_support(rec['name'], chunks))
        if rec['description']:
            trace_present += 1
            trace_scores.append(_best_support(rec['description'], chunks))
        else:
            trace_scores.append(0.0)
    trace_present_rate = trace_present / float(len(records))
    label_support_mean = sum(label_scores) / float(len(label_scores)) if label_scores else 0.0
    trace_support_mean = sum(trace_scores) / float(len(trace_scores)) if trace_scores else 0.0
    evidence_score = 0.2 * trace_present_rate + 0.4 * label_support_mean + 0.4 * trace_support_mean
    return {
        'trace_present_rate': trace_present_rate,
        'label_support_mean': label_support_mean,
        'trace_support_mean': trace_support_mean,
        'evidence_score': evidence_score,
    }
