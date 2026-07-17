from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


Row = Dict[str, Any]


def make_comparison_row(dataset: str, pipeline: str, variant: str, metrics: Dict[str, Any]) -> Row:
    row = {'dataset': dataset, 'pipeline': pipeline, 'variant': variant}
    row.update(metrics)
    return row


def make_baseline_row(dataset: str, pipeline: str, selector: str, metrics: Dict[str, Any]) -> Row:
    row = {'dataset': dataset, 'pipeline': pipeline, 'selector': selector}
    row.update(metrics)
    return row


def make_ablation_row(dataset: str, pipeline: str, ablation: str, metrics: Dict[str, Any]) -> Row:
    row = {'dataset': dataset, 'pipeline': pipeline, 'ablation': ablation}
    row.update(metrics)
    return row


def _fieldnames(rows: Iterable[Row]) -> List[str]:
    seen: List[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    return seen


def write_csv(rows: List[Row], path: str | Path) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with out_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: List[Row], path: str | Path) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')


def write_markdown_table(rows: List[Row], path: str | Path) -> None:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text('\n'.join(['| empty |', '|---|', '| no data |']) + '\n', encoding='utf-8')
        return
    headers = _fieldnames(rows)
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(h, '')) for h in headers) + ' |')
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
