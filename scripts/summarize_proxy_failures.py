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

from pathlib import Path

from fame.evaluation.proxy_compare import load_ranking_csv
_proxy_reporting = _load_local_module('proxy_reporting')
write_csv = _proxy_reporting.write_csv
write_json = _proxy_reporting.write_json


def main() -> None:
    ap = argparse.ArgumentParser(description='Summarize proxy admissibility/validation failures.')
    ap.add_argument('--proxy-csv', required=True)
    ap.add_argument('--dataset', default='')
    ap.add_argument('--pipeline', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    rows = load_ranking_csv(args.proxy_csv)
    summary = {
        'dataset': args.dataset,
        'pipeline': args.pipeline,
        'total_candidates': len(rows),
        'eligible_candidates': sum(1 for row in rows if bool(row.get('eligible_ok'))),
        'xsd_failures': sum(1 for row in rows if not bool(row.get('wellformed_ok', True))),
        'duplicate_failures': sum(1 for row in rows if not bool(row.get('duplicate_free', row.get('has_duplicate_features') is False))),
        'sat_failures': sum(1 for row in rows if row.get('satisfiable') is False),
        'constraint_failures': sum(1 for row in rows if not bool(row.get('constraints_valid', True))),
        'parse_errors': sum(1 for row in rows if row.get('error') not in (None, '')),
    }

    print(f"Total candidates : {summary['total_candidates']}")
    print(f"Eligible         : {summary['eligible_candidates']}")
    print(f"XSD failures     : {summary['xsd_failures']}")
    print(f"Duplicate issues : {summary['duplicate_failures']}")
    print(f"SAT failures     : {summary['sat_failures']}")
    print(f"Constraint issues: {summary['constraint_failures']}")
    print(f"Parse errors     : {summary['parse_errors']}")

    if args.out:
        if args.json or Path(args.out).suffix.lower() == '.json':
            write_json([summary], args.out)
        else:
            write_csv([summary], args.out)


if __name__ == '__main__':
    main()
