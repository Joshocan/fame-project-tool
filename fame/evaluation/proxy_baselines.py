from __future__ import annotations

import random
from typing import Callable, Dict, List


Row = Dict[str, object]


def _eligible_rows(rows: List[Row]) -> List[Row]:
    eligible = [row for row in rows if bool(row.get('eligible_ok'))]
    return eligible or rows


def _numeric(row: Row, key: str) -> float:
    value = row.get(key)
    return float(value) if value not in (None, '') else float('-inf')


def select_first_candidate(rows: List[Row]) -> Row:
    if not rows:
        raise ValueError('No candidate rows provided.')
    return rows[0]


def select_random_admissible(rows: List[Row], seed: int = 42) -> Row:
    candidates = _eligible_rows(rows)
    rng = random.Random(seed)
    return rng.choice(candidates)


def select_best_evidence(rows: List[Row]) -> Row:
    candidates = _eligible_rows(rows)
    return max(candidates, key=lambda row: (_numeric(row, 'evidence_score'), -int(row.get('rank', 10**9))))


def select_best_consensus(rows: List[Row]) -> Row:
    candidates = _eligible_rows(rows)
    return max(candidates, key=lambda row: (_numeric(row, 'consensus_score'), -int(row.get('rank', 10**9))))


def select_best_proxy(rows: List[Row]) -> Row:
    candidates = _eligible_rows(rows)
    return max(candidates, key=lambda row: (_numeric(row, 'proxy_score'), -int(row.get('rank', 10**9))))


def get_selector(name: str) -> Callable[..., Row]:
    normalized = name.strip().lower()
    mapping = {
        'first': select_first_candidate,
        'random_admissible': select_random_admissible,
        'random': select_random_admissible,
        'evidence_only': select_best_evidence,
        'consensus_only': select_best_consensus,
        'proxy': select_best_proxy,
    }
    if normalized not in mapping:
        raise KeyError(f"Unknown selector: {name}")
    return mapping[normalized]
