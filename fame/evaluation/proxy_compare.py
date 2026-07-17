from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


Row = Dict[str, object]


def _coerce_value(value: str) -> object:
    lowered = value.strip().lower()
    if lowered in {'true', 'false'}:
        return lowered == 'true'
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_ranking_csv(path: str | Path) -> List[Row]:
    csv_path = Path(path).expanduser().resolve()
    with csv_path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append({key: _coerce_value(value) for key, value in row.items()})
    return rows


def _rank_map(rows: Sequence[Row]) -> Dict[str, int]:
    rank_map: Dict[str, int] = {}
    for idx, row in enumerate(rows, start=1):
        fm_xml = str(row.get('fm_xml') or '')
        if not fm_xml:
            continue
        rank = row.get('rank')
        rank_map[fm_xml] = int(rank) if rank not in (None, '') else idx
    return rank_map


def compute_top1_agreement(proxy_rows: Sequence[Row], gt_rows: Sequence[Row]) -> bool:
    if not proxy_rows or not gt_rows:
        return False
    return str(proxy_rows[0].get('fm_xml')) == str(gt_rows[0].get('fm_xml'))


def compute_topk_overlap(proxy_rows: Sequence[Row], gt_rows: Sequence[Row], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    proxy_set = {str(row.get('fm_xml')) for row in proxy_rows[:k] if row.get('fm_xml')}
    gt_set = {str(row.get('fm_xml')) for row in gt_rows[:k] if row.get('fm_xml')}
    if not proxy_set and not gt_set:
        return 0.0
    return len(proxy_set & gt_set) / float(k)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float('nan')


def compute_spearman(proxy_rows: Sequence[Row], gt_rows: Sequence[Row]) -> float | None:
    proxy_map = _rank_map(proxy_rows)
    gt_map = _rank_map(gt_rows)
    common = sorted(set(proxy_map) & set(gt_map))
    n = len(common)
    if n < 2:
        return None
    diffs = [(proxy_map[item] - gt_map[item]) ** 2 for item in common]
    return 1.0 - (6.0 * sum(diffs)) / (n * (n * n - 1.0))


def compute_kendall_tau(proxy_rows: Sequence[Row], gt_rows: Sequence[Row]) -> float | None:
    proxy_map = _rank_map(proxy_rows)
    gt_map = _rank_map(gt_rows)
    common = sorted(set(proxy_map) & set(gt_map))
    n = len(common)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = common[i]
            b = common[j]
            proxy_diff = proxy_map[a] - proxy_map[b]
            gt_diff = gt_map[a] - gt_map[b]
            product = proxy_diff * gt_diff
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    denom = concordant + discordant
    if denom == 0:
        return None
    return (concordant - discordant) / float(denom)


def compare_rankings(proxy_rows: Sequence[Row], gt_rows: Sequence[Row], ks: tuple[int, ...] = (1, 3, 5)) -> Dict[str, object]:
    proxy_map = _rank_map(proxy_rows)
    gt_map = _rank_map(gt_rows)
    result: Dict[str, object] = {
        'proxy_top1': str(proxy_rows[0].get('fm_xml')) if proxy_rows else '',
        'gt_top1': str(gt_rows[0].get('fm_xml')) if gt_rows else '',
        'top1_match': compute_top1_agreement(proxy_rows, gt_rows),
        'spearman_rho': compute_spearman(proxy_rows, gt_rows),
        'kendall_tau': compute_kendall_tau(proxy_rows, gt_rows),
        'common_candidates': len(set(proxy_map) & set(gt_map)),
        'proxy_candidates': len(proxy_map),
        'gt_candidates': len(gt_map),
    }
    for k in ks:
        result[f'top{k}_overlap'] = compute_topk_overlap(proxy_rows, gt_rows, k=k)
    return result
