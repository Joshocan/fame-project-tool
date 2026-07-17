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
    spec.loader.exec_module(module)
    return module

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

_proxy_compare = _load_local_module('proxy_compare')
compare_rankings = _proxy_compare.compare_rankings
load_ranking_csv = _proxy_compare.load_ranking_csv
_proxy_reporting = _load_local_module('proxy_reporting')
make_ablation_row = _proxy_reporting.make_ablation_row
write_csv = _proxy_reporting.write_csv


VARIANTS = {
    'full_proxy': {'use_evidence': True, 'use_consensus': True, 'require_sat': False},
    'evidence_only': {'use_evidence': True, 'use_consensus': False, 'require_sat': False},
    'consensus_only': {'use_evidence': False, 'use_consensus': True, 'require_sat': False},
    'no_sat': {'use_evidence': True, 'use_consensus': True, 'require_sat': False},
    'with_sat': {'use_evidence': True, 'use_consensus': True, 'require_sat': True},
}


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description='Run proxy selector ablation variants.')
    ap.add_argument('--fm-dir', required=True)
    ap.add_argument('--chunks-dir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--gt-csv', default='')
    ap.add_argument('--xsd-path', default='')
    ap.add_argument('--dataset', default='')
    ap.add_argument('--pipeline', default='')
    ap.add_argument('--top-fm', type=int, default=5)
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS))
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for variant in args.variants:
        if variant not in VARIANTS:
            raise KeyError(f'Unknown ablation variant: {variant}')
        spec = VARIANTS[variant]
        variant_out = outdir / variant
        cmd = [
            sys.executable,
            'scripts/rank_proxy_fm.py',
            '--fm-dir', args.fm_dir,
            '--chunks-dir', args.chunks_dir,
            '--outdir', str(variant_out),
            '--top-fm', str(args.top_fm),
            '--selector-use-evidence', str(spec['use_evidence']).lower(),
            '--selector-use-consensus', str(spec['use_consensus']).lower(),
        ]
        if args.xsd_path:
            cmd.extend(['--xsd-path', args.xsd_path])
        if spec['require_sat']:
            cmd.append('--require-sat')
        _run(cmd)
        proxy_csv = variant_out / 'proxy_fm' / f'top_{args.top_fm}' / 'proxy_scores.csv'
        proxy_rows = load_ranking_csv(proxy_csv)
        metrics: Dict[str, object] = {'eligible_count': sum(1 for row in proxy_rows if row.get('eligible_ok'))}
        if args.gt_csv:
            gt_rows = load_ranking_csv(args.gt_csv)
            metrics.update(compare_rankings(proxy_rows, gt_rows))
        selected = proxy_rows[0] if proxy_rows else {}
        metrics.update({
            'selected_fm': selected.get('fm_xml', ''),
            'evidence_score': selected.get('evidence_score'),
            'consensus_score': selected.get('consensus_score'),
            'proxy_score': selected.get('proxy_score'),
        })
        rows.append(make_ablation_row(args.dataset, args.pipeline, variant, metrics))

    write_csv(rows, outdir / 'ablation_summary.csv')
    print(f'Wrote ablation summary to {outdir / "ablation_summary.csv"}')


if __name__ == '__main__':
    main()
