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
from typing import Any, Dict, List

_proxy_baselines = _load_local_module('proxy_baselines')
get_selector = _proxy_baselines.get_selector
from fame.evaluation.proxy_compare import load_ranking_csv
_proxy_reporting = _load_local_module('proxy_reporting')
make_baseline_row = _proxy_reporting.make_baseline_row
write_csv = _proxy_reporting.write_csv
write_json = _proxy_reporting.write_json


def _safe_bool(value: Any) -> bool | None:
    if value in (None, ''):
        return None
    return bool(value)


def _evaluate_selected_fm(selected_fm: Path, gt_path: Path, model_name: str, threshold: float, top_k: int, feature_weight: float, parent_weight: float) -> Dict[str, Any]:
    from fame.evaluation.coverage import CoverageConfig, CoverageEvaluator
    from fame.evaluation.quality_sat import analyze_sat_quality
    from fame.evaluation.semantic import semantic_prf
    from fame.evaluation.wellformed import validate_feature_model

    cfg = CoverageConfig(
        model_name=model_name,
        similarity_threshold=threshold,
        top_k=top_k,
        feature_weight=feature_weight,
        parent_weight=parent_weight,
    )
    evaluator = CoverageEvaluator(cfg)
    coverage_score = evaluator.score(gt_path, selected_fm, verbose=False)
    sem = semantic_prf(gt_path, selected_fm, model=evaluator.model, threshold=threshold)
    wf = validate_feature_model(selected_fm, xsd_path=None)
    sat = analyze_sat_quality(selected_fm, compute_products=False)
    return {
        'semantic_precision': sem.get('semantic_precision'),
        'semantic_recall': sem.get('semantic_recall'),
        'semantic_f1': sem.get('semantic_f1'),
        'coverage': coverage_score,
        'wellformed_ok': wf.ok,
        'satisfiable': sat.satisfiable,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Evaluate baseline selection strategies against GT metrics.')
    ap.add_argument('--candidate-csv', required=True)
    ap.add_argument('--gt-path', required=True)
    ap.add_argument('--dataset', default='')
    ap.add_argument('--pipeline', default='')
    ap.add_argument('--selectors', nargs='+', default=['proxy', 'first', 'random_admissible', 'evidence_only', 'consensus_only'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--coverage-model-name', default='sentence-transformers/all-mpnet-base-v2')
    ap.add_argument('--coverage-threshold', type=float, default=0.55)
    ap.add_argument('--coverage-top-k', type=int, default=5)
    ap.add_argument('--coverage-feature-weight', type=float, default=0.7)
    ap.add_argument('--coverage-parent-weight', type=float, default=0.3)
    ap.add_argument('--out', default='')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    rows = load_ranking_csv(args.candidate_csv)
    gt_path = Path(args.gt_path).expanduser().resolve()
    result_rows: List[Dict[str, Any]] = []
    for selector_name in args.selectors:
        selector = get_selector(selector_name)
        if selector_name in {'random_admissible', 'random'}:
            selected = selector(rows, seed=args.seed)
        else:
            selected = selector(rows)
        fm_path = Path(str(selected.get('fm_xml'))).expanduser().resolve()
        metrics = _evaluate_selected_fm(
            fm_path,
            gt_path,
            args.coverage_model_name,
            args.coverage_threshold,
            args.coverage_top_k,
            args.coverage_feature_weight,
            args.coverage_parent_weight,
        )
        result_rows.append(make_baseline_row(args.dataset, args.pipeline, selector_name, {
            'selected_fm': str(fm_path),
            'eligible_ok': _safe_bool(selected.get('eligible_ok')),
            'proxy_score': selected.get('proxy_score'),
            'evidence_score': selected.get('evidence_score'),
            'consensus_score': selected.get('consensus_score'),
            **metrics,
        }))

    if args.out:
        if args.json or Path(args.out).suffix.lower() == '.json':
            write_json(result_rows, args.out)
        else:
            write_csv(result_rows, args.out)
    else:
        for row in result_rows:
            print(row)


if __name__ == '__main__':
    main()
