#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util


def _load_local_module(name: str):
    module_path = ROOT / 'fame' / 'evaluation' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'paper2_{name}', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module: {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

_proxy_reporting = _load_local_module('proxy_reporting')
write_csv = _proxy_reporting.write_csv
write_markdown_table = _proxy_reporting.write_markdown_table


Row = Dict[str, Any]


def _load_csv(path: Path) -> List[Row]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def _mean(values: Iterable[Any]) -> float:
    nums = [float(v) for v in values if v not in (None, '', 'None')]
    return sum(nums) / len(nums) if nums else float('nan')


def _summarize(rows: List[Row], group_keys: List[str], metrics: List[str]) -> List[Row]:
    grouped: Dict[tuple, List[Row]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, '') for key in group_keys)].append(row)
    summaries: List[Row] = []
    for group, items in grouped.items():
        summary = {key: value for key, value in zip(group_keys, group)}
        summary['count'] = len(items)
        for metric in metrics:
            summary[f'mean_{metric}'] = _mean(item.get(metric) for item in items)
        summaries.append(summary)
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description='Aggregate Paper 2 experiment result CSVs.')
    ap.add_argument('--results-root', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    root = Path(args.results_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    comparison_rows: List[Row] = []
    baseline_rows: List[Row] = []
    ablation_rows: List[Row] = []
    for path in root.rglob('*.csv'):
        name = path.name
        rows = _load_csv(path)
        if name == 'comparison_summary.csv':
            comparison_rows.extend(rows)
        elif name == 'baseline_summary.csv':
            baseline_rows.extend(rows)
        elif name == 'ablation_summary.csv':
            ablation_rows.extend(rows)

    if comparison_rows:
        summary = _summarize(comparison_rows, ['dataset', 'pipeline'], ['top1_match', 'top3_overlap', 'top5_overlap', 'spearman_rho'])
        write_csv(summary, outdir / 'paper2_comparison_summary.csv')
    if baseline_rows:
        summary = _summarize(baseline_rows, ['dataset', 'pipeline', 'selector'], ['semantic_f1', 'coverage', 'satisfiable'])
        write_csv(summary, outdir / 'paper2_baseline_summary.csv')
    if ablation_rows:
        summary = _summarize(ablation_rows, ['dataset', 'pipeline', 'ablation'], ['top1_match', 'top5_overlap', 'spearman_rho', 'eligible_count'])
        write_csv(summary, outdir / 'paper2_ablation_summary.csv')
    md_rows = []
    for csv_name in ['paper2_comparison_summary.csv', 'paper2_baseline_summary.csv', 'paper2_ablation_summary.csv']:
        path = outdir / csv_name
        if path.exists():
            md_rows.append({'table': csv_name, 'path': str(path)})
    write_markdown_table(md_rows, outdir / 'paper2_tables.md')
    print(f'Wrote aggregated results to {outdir}')


if __name__ == '__main__':
    main()
