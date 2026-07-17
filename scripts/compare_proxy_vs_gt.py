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

import json
from pathlib import Path

_proxy_compare = _load_local_module('proxy_compare')
compare_rankings = _proxy_compare.compare_rankings
load_ranking_csv = _proxy_compare.load_ranking_csv
_proxy_reporting = _load_local_module('proxy_reporting')
make_comparison_row = _proxy_reporting.make_comparison_row
write_csv = _proxy_reporting.write_csv
write_json = _proxy_reporting.write_json


def main() -> None:
    ap = argparse.ArgumentParser(description='Compare proxy ranking against GT-based ranking.')
    ap.add_argument('--proxy-csv', required=True)
    ap.add_argument('--gt-csv', required=True)
    ap.add_argument('--dataset', default='')
    ap.add_argument('--pipeline', default='')
    ap.add_argument('--variant', default='proxy')
    ap.add_argument('--out', default='')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    proxy_rows = load_ranking_csv(args.proxy_csv)
    gt_rows = load_ranking_csv(args.gt_csv)
    metrics = compare_rankings(proxy_rows, gt_rows)

    print(f"Dataset      : {args.dataset or '-'}")
    print(f"Pipeline     : {args.pipeline or '-'}")
    print(f"Variant      : {args.variant}")
    print(f"Top-1 match  : {metrics['top1_match']}")
    for key in ('top3_overlap', 'top5_overlap', 'spearman_rho', 'kendall_tau'):
        if key in metrics:
            print(f"{key:13}: {metrics[key]}")

    if args.out:
        row = make_comparison_row(args.dataset, args.pipeline, args.variant, metrics)
        out_path = Path(args.out)
        if args.json or out_path.suffix.lower() == '.json':
            write_json([row], out_path)
        else:
            write_csv([row], out_path)
    elif args.json:
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
