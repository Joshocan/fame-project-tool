from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

from .constraints import extract_constraints
from .feature_list import extract_feature_list


@dataclass(frozen=True)
class ProxyConsensusConfig:
    feature_weight: float = 0.6
    edge_weight: float = 0.4


@dataclass(frozen=True)
class ProxyConsensusResult:
    feature_overlap_mean: float
    edge_overlap_mean: float
    constraint_overlap_mean: float
    consensus_score: float


@dataclass(frozen=True)
class CandidateSignature:
    features: frozenset[str]
    edges: frozenset[tuple[str, str]]
    constraints: frozenset[tuple[str, tuple[str, ...], tuple[str, ...]]]


def _norm(text: str) -> str:
    return ' '.join((text or '').strip().lower().split())


def _jaccard(left: Set, right: Set) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def _extract_edges(xml_path: Path | str) -> Set[Tuple[str, str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    struct = root.find('struct')
    if struct is None:
        return set()
    edges: Set[Tuple[str, str]] = set()

    def walk(node: ET.Element, parent_name: str | None = None) -> None:
        name = (node.attrib.get('name') or '').strip()
        current_parent = parent_name
        if name:
            if parent_name:
                edges.add((_norm(parent_name), _norm(name)))
            current_parent = name
        for child in node:
            if child.tag in {'feature', 'and', 'or', 'alt'}:
                walk(child, current_parent)

    for child in struct:
        if child.tag in {'feature', 'and', 'or', 'alt'}:
            walk(child)
    return edges


def build_candidate_signature(xml_path: Path | str) -> CandidateSignature:
    xml_path = Path(xml_path)
    features = frozenset(_norm(rec.feature_name) for rec in extract_feature_list(xml_path) if rec.feature_name)
    edges = frozenset(_extract_edges(xml_path))
    constraints = frozenset(
        (
            _norm(rec.constraint_type),
            tuple(sorted(_norm(x) for x in rec.left_features if _norm(x))),
            tuple(sorted(_norm(x) for x in rec.right_features if _norm(x))),
        )
        for rec in extract_constraints(xml_path)
    )
    return CandidateSignature(features=features, edges=edges, constraints=constraints)


def score_consensus(signature: CandidateSignature, peers: Sequence[CandidateSignature], cfg: ProxyConsensusConfig) -> ProxyConsensusResult:
    if not peers:
        return ProxyConsensusResult(1.0, 1.0, 1.0, 1.0)

    feature_scores = [_jaccard(set(signature.features), set(peer.features)) for peer in peers]
    edge_scores = [_jaccard(set(signature.edges), set(peer.edges)) for peer in peers]
    constraint_scores = [_jaccard(set(signature.constraints), set(peer.constraints)) for peer in peers]

    feature_overlap_mean = sum(feature_scores) / len(feature_scores)
    edge_overlap_mean = sum(edge_scores) / len(edge_scores)
    constraint_overlap_mean = sum(constraint_scores) / len(constraint_scores)
    consensus_score = float(cfg.feature_weight) * feature_overlap_mean + float(cfg.edge_weight) * edge_overlap_mean
    consensus_score = max(0.0, min(1.0, consensus_score))

    return ProxyConsensusResult(
        feature_overlap_mean=round(feature_overlap_mean, 6),
        edge_overlap_mean=round(edge_overlap_mean, 6),
        constraint_overlap_mean=round(constraint_overlap_mean, 6),
        consensus_score=round(consensus_score, 6),
    )
