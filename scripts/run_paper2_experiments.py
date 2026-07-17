#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def _load_cfg(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError('PyYAML is required for run_paper2_experiments.py') from exc
    data = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Invalid config file: {cfg_path}')
    return data


def _run(cmd: list[str]) -> None:
    print('[paper2]', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description='Run end-to-end Paper 2 experiments from YAML config.')
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    run_cfg = cfg.get('run', {})
    results_root = Path(run_cfg.get('results_root', 'results/paper2')).expanduser().resolve()
    top_k = int(run_cfg.get('top_k', 5))

    for dataset in cfg.get('datasets', []):
        name = dataset['name']
        pipeline = dataset.get('pipeline', 'default')
        dataset_root = results_root / name / pipeline
        dataset_root.mkdir(parents=True, exist_ok=True)

        gt_top_out = dataset_root / 'top_fm'
        _run([
            sys.executable, 'scripts/rank_top_fm.py',
            '--fm-dir', dataset['fm_dir'],
            '--outdir', str(gt_top_out),
            '--gt-path', dataset['gt_path'],
            '--top-fm', str(top_k),
            '--glob', dataset.get('glob', '*.xml'),
        ] + (['--xsd-path', dataset['xsd_path']] if dataset.get('xsd_path') else []) + (['--require-sat'] if run_cfg.get('require_sat') else []))

        proxy_out = dataset_root / 'proxy_fm'
        _run([
            sys.executable, 'scripts/rank_proxy_fm.py',
            '--fm-dir', dataset['fm_dir'],
            '--chunks-dir', dataset['chunks_dir'],
            '--outdir', str(proxy_out),
            '--top-fm', str(top_k),
        ] + (['--xsd-path', dataset['xsd_path']] if dataset.get('xsd_path') else []) + (['--require-sat'] if run_cfg.get('require_sat') else []))

        gt_csv = gt_top_out / 'top_fm' / f'top_{top_k}' / 'top_fm_scores.csv'
        proxy_csv = proxy_out / 'proxy_fm' / f'top_{top_k}' / 'proxy_scores.csv'
        comparison_out = dataset_root / 'comparisons' / 'comparison_summary.csv'
        _run([
            sys.executable, 'scripts/compare_proxy_vs_gt.py',
            '--proxy-csv', str(proxy_csv),
            '--gt-csv', str(gt_csv),
            '--dataset', name,
            '--pipeline', pipeline,
            '--out', str(comparison_out),
        ])

        baseline_out = dataset_root / 'baselines' / 'baseline_summary.csv'
        _run([
            sys.executable, 'scripts/evaluate_selection_baselines.py',
            '--candidate-csv', str(proxy_csv),
            '--gt-path', dataset['gt_path'],
            '--dataset', name,
            '--pipeline', pipeline,
            '--out', str(baseline_out),
        ])

        ablation_out = dataset_root / 'ablations'
        _run([
            sys.executable, 'scripts/run_proxy_ablation.py',
            '--fm-dir', dataset['fm_dir'],
            '--chunks-dir', dataset['chunks_dir'],
            '--outdir', str(ablation_out),
            '--gt-csv', str(gt_csv),
            '--dataset', name,
            '--pipeline', pipeline,
            '--top-fm', str(top_k),
        ] + (['--xsd-path', dataset['xsd_path']] if dataset.get('xsd_path') else []))

    _run([
        sys.executable, 'scripts/aggregate_paper2_results.py',
        '--results-root', str(results_root),
        '--outdir', str(results_root / 'summary'),
    ])


if __name__ == '__main__':
    main()
