from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
import xml.etree.ElementTree as ET


def normalize_name(name: str) -> str:
    return ' '.join((name or '').replace('_', ' ').replace('-', ' ').lower().split())


def extract_signature(xml_path: str | Path) -> Dict[str, object]:
    xml_path = Path(xml_path).expanduser().resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()
    features: Set[str] = set()
    edges: Set[Tuple[str, str]] = set()
    constraints: Set[str] = set()

    def walk(elem: ET.Element, parent_name: str | None = None) -> None:
        current_name = (elem.attrib.get('name') or elem.attrib.get('feature') or '').strip()
        normalized = normalize_name(current_name) if current_name else None
        if normalized:
            features.add(normalized)
            if parent_name:
                edges.add((parent_name, normalized))
            parent_name = normalized
        for child in list(elem):
            if child.tag.lower().endswith('constraints'):
                raw = ''.join(child.itertext()).strip()
                if raw:
                    constraints.add(' '.join(raw.split()))
            walk(child, parent_name)

    walk(root)
    return {'features': features, 'edges': edges, 'constraints': constraints}


def jaccard(a: Set[object], b: Set[object]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / float(len(union))


def score_consensus(target: Dict[str, object], others: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not others:
        return {
            'feature_overlap_mean': 1.0,
            'edge_overlap_mean': 1.0,
            'constraint_overlap_mean': 1.0,
            'consensus_score': 1.0,
        }
    feature_scores = [jaccard(target['features'], other['features']) for other in others]
    edge_scores = [jaccard(target['edges'], other['edges']) for other in others]
    constraint_scores = [jaccard(target['constraints'], other['constraints']) for other in others]
    feature_overlap_mean = sum(feature_scores) / len(feature_scores)
    edge_overlap_mean = sum(edge_scores) / len(edge_scores)
    constraint_overlap_mean = sum(constraint_scores) / len(constraint_scores)
    consensus_score = 0.6 * feature_overlap_mean + 0.4 * edge_overlap_mean
    return {
        'feature_overlap_mean': feature_overlap_mean,
        'edge_overlap_mean': edge_overlap_mean,
        'constraint_overlap_mean': constraint_overlap_mean,
        'consensus_score': consensus_score,
    }
