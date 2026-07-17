from __future__ import annotations

import csv
import importlib.util
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import xml.etree.ElementTree as ET


THIS_DIR = Path(__file__).resolve().parent


def _load_sibling(name: str):
    path = THIS_DIR / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'paper2_{name}', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load sibling module: {path}')
    import sys
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_proxy_evidence = _load_sibling('proxy_evidence')
_proxy_consensus = _load_sibling('proxy_consensus')

score_evidence = _proxy_evidence.score_evidence
extract_signature = _proxy_consensus.extract_signature
score_consensus = _proxy_consensus.score_consensus


@dataclass(frozen=True)
class ProxyRankConfig:
    top_n: int
    chunks_dir: Path
    xsd_path: Optional[Path] = None
    output_subdir: str = ''
    require_sat: bool = False
    use_evidence: bool = True
    use_consensus: bool = True
    evidence_weight: float = 0.7
    consensus_weight: float = 0.3


def _load_external(module_name: str, rel_path: str, attr: str):
    try:
        spec = importlib.util.spec_from_file_location(module_name, THIS_DIR / rel_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, attr, None)
    except Exception:
        return None


def _validate_with_external(xml_path: Path, xsd_path: Optional[Path]) -> Dict[str, Any]:
    validate_fn = _load_external('paper2_wellformed', 'wellformed.py', 'validate_feature_model')
    if validate_fn is None:
        try:
            ET.parse(xml_path)
            return {'ok': True, 'errors': []}
        except Exception as exc:
            return {'ok': False, 'errors': [str(exc)]}
    try:
        result = validate_fn(xml_path, xsd_path=xsd_path)
        return {'ok': bool(getattr(result, 'ok', False)), 'errors': list(getattr(result, 'errors', []))}
    except Exception as exc:
        return {'ok': False, 'errors': [str(exc)]}


def _analyze_sat(xml_path: Path) -> tuple[Optional[bool], str]:
    sat_fn = _load_external('paper2_quality_sat', 'quality_sat.py', 'analyze_sat_quality')
    if sat_fn is None:
        return None, ''
    try:
        result = sat_fn(xml_path, compute_products=False)
        return bool(getattr(result, 'satisfiable', False)), ''
    except Exception as exc:
        return None, str(exc)


def _extract_names(xml_path: Path) -> List[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    names: List[str] = []
    for elem in root.iter():
        name = (elem.attrib.get('name') or elem.attrib.get('feature') or '').strip()
        if name:
            names.append(name)
    return names


def check_admissibility(xml_path: str | Path, xsd_path: Optional[Path], require_sat: bool = False) -> Dict[str, Any]:
    xml_path = Path(xml_path).expanduser().resolve()
    wf = _validate_with_external(xml_path, xsd_path)
    duplicate_names: List[str] = []
    duplicate_error = ''
    try:
        counts = Counter(_extract_names(xml_path))
        duplicate_names = sorted(name for name, count in counts.items() if count > 1)
    except Exception as exc:
        duplicate_error = str(exc)
    duplicate_free = not duplicate_names and not duplicate_error
    constraints_valid = True
    constraints_error = ''
    satisfiable: Optional[bool] = None
    sat_error = ''
    if wf['ok'] and duplicate_free:
        satisfiable, sat_error = _analyze_sat(xml_path)
    eligible_ok = bool(wf['ok']) and duplicate_free
    if require_sat:
        eligible_ok = eligible_ok and bool(satisfiable) and not sat_error
    return {
        'wellformed_ok': bool(wf['ok']),
        'wellformed_error_count': len(wf['errors']),
        'wellformed_errors': wf['errors'],
        'duplicate_free': duplicate_free,
        'has_duplicate_features': not duplicate_free,
        'duplicate_feature_count': len(duplicate_names),
        'duplicate_feature_names': duplicate_names,
        'duplicate_feature_error': duplicate_error,
        'constraints_valid': constraints_valid,
        'constraints_error': constraints_error,
        'satisfiable': satisfiable,
        'sat_error': sat_error,
        'eligible_ok': eligible_ok,
    }


def _summary_sort_key(row: Dict[str, Any]) -> tuple:
    return (
        1 if row.get('eligible_ok') else 0,
        float(row.get('proxy_score') or -1),
        float(row.get('evidence_score') or -1),
        float(row.get('consensus_score') or -1),
        -int(row.get('candidate_index') or 0),
    )


def build_proxy_rows(*, candidates: Sequence[Dict[str, Any]], cfg: ProxyRankConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eligible_indices: List[int] = []
    signatures: List[Optional[Dict[str, object]]] = []
    total = len(candidates)
    print(f'[proxy_selector] Evaluating {total} candidate FM files...')
    for idx, candidate in enumerate(candidates, start=1):
        fm_xml = Path(str(candidate.get('fm_xml') or candidate.get('final_xml') or '')).expanduser().resolve()
        run_id = str(candidate.get('run_id') or fm_xml.stem)
        print(f'[proxy_selector] [{idx}/{total}] Checking {run_id}')
        if not fm_xml.exists():
            continue
        admissibility = check_admissibility(fm_xml, cfg.xsd_path, require_sat=cfg.require_sat)
        row: Dict[str, Any] = {
            'candidate_index': idx,
            'run_id': run_id,
            'fm_xml': str(fm_xml),
            'meta': candidate.get('meta'),
            **admissibility,
            'trace_present_rate': None,
            'label_support_mean': None,
            'trace_support_mean': None,
            'evidence_score': None,
            'feature_overlap_mean': None,
            'edge_overlap_mean': None,
            'constraint_overlap_mean': None,
            'consensus_score': None,
            'proxy_score': None,
        }
        signature = None
        if row['eligible_ok']:
            if cfg.use_evidence:
                try:
                    row.update(score_evidence(fm_xml, cfg.chunks_dir))
                except Exception as exc:
                    row['evidence_error'] = str(exc)
                    row['eligible_ok'] = False
            if row['eligible_ok']:
                try:
                    signature = extract_signature(fm_xml)
                except Exception as exc:
                    row['consensus_error'] = str(exc)
            eligible_indices.append(len(rows))
        else:
            reasons = []
            if not row['wellformed_ok']:
                reasons.append(f"xsd_or_wellformed_errors={row['wellformed_error_count']}")
            if not row['duplicate_free']:
                reasons.append('duplicate_check_failed')
            if cfg.require_sat and row.get('satisfiable') is False:
                reasons.append('unsatisfiable')
            if cfg.require_sat and row.get('sat_error'):
                reasons.append('sat_check_failed')
            print(f"[proxy_selector] [{idx}/{total}] Ineligible: {', '.join(reasons) if reasons else 'unknown reason'}")
        rows.append(row)
        signatures.append(signature)

    gate_total = len(rows)
    print('[proxy_selector] Gate summary:')
    print(f"[proxy_selector]   Parsed/XSD-valid      : {sum(1 for r in rows if r.get('wellformed_ok'))}/{gate_total}")
    print(f"[proxy_selector]   Duplicate-free        : {sum(1 for r in rows if r.get('duplicate_free'))}/{gate_total}")
    print(f"[proxy_selector]   Constraints-valid     : {sum(1 for r in rows if r.get('constraints_valid'))}/{gate_total}")
    print(f"[proxy_selector]   Satisfiable           : {sum(1 for r in rows if r.get('satisfiable') is True)}/{gate_total}")
    print(f"[proxy_selector]   Passed all hard gates : {sum(1 for r in rows if r.get('eligible_ok'))}/{gate_total}")

    eligible_sigs = [(i, sig) for i, sig in enumerate(signatures) if sig is not None and rows[i].get('eligible_ok')]
    for row_index, sig in eligible_sigs:
        if cfg.use_consensus:
            others = [other_sig for other_idx, other_sig in eligible_sigs if other_idx != row_index]
            rows[row_index].update(score_consensus(sig, others))
        else:
            rows[row_index].update({
                'feature_overlap_mean': 0.0,
                'edge_overlap_mean': 0.0,
                'constraint_overlap_mean': 0.0,
                'consensus_score': 0.0,
            })
        evidence_score = float(rows[row_index].get('evidence_score') or 0.0) if cfg.use_evidence else 0.0
        consensus_score = float(rows[row_index].get('consensus_score') or 0.0) if cfg.use_consensus else 0.0
        if cfg.use_evidence and cfg.use_consensus:
            proxy_score = cfg.evidence_weight * evidence_score + cfg.consensus_weight * consensus_score
        elif cfg.use_evidence:
            proxy_score = evidence_score
        elif cfg.use_consensus:
            proxy_score = consensus_score
        else:
            proxy_score = 0.0
        rows[row_index]['proxy_score'] = proxy_score

    return sorted(rows, key=_summary_sort_key, reverse=True)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def rank_proxy_fms(*, candidates: Sequence[Dict[str, Any]], pipeline_root: Path, cfg: ProxyRankConfig) -> Optional[Dict[str, Any]]:
    if cfg.top_n <= 0 or not candidates:
        return None
    ranked_rows = build_proxy_rows(candidates=candidates, cfg=cfg)
    top_root = pipeline_root / 'proxy_fm'
    if cfg.output_subdir:
        top_root = top_root / cfg.output_subdir
    top_root.mkdir(parents=True, exist_ok=True)

    summary_csv = top_root / 'proxy_scores.csv'
    fieldnames = [
        'candidate_index', 'run_id', 'fm_xml', 'eligible_ok', 'wellformed_ok', 'duplicate_free',
        'constraints_valid', 'satisfiable', 'trace_present_rate', 'label_support_mean',
        'trace_support_mean', 'evidence_score', 'feature_overlap_mean', 'edge_overlap_mean',
        'constraint_overlap_mean', 'consensus_score', 'proxy_score', 'duplicate_feature_count',
        'wellformed_error_count', 'sat_error'
    ]
    rows_with_rank = []
    for rank, row in enumerate(ranked_rows, start=1):
        copy = dict(row)
        copy['rank'] = rank
        rows_with_rank.append(copy)
    _write_csv(summary_csv, rows_with_rank, ['rank'] + fieldnames)

    eligible_rows = [row for row in ranked_rows if row.get('eligible_ok')]
    print(f'[proxy_selector] Total candidates passing hard gates: {len(eligible_rows)}/{len(ranked_rows)}')
    if not eligible_rows:
        print('[proxy_selector] No candidates passed the hard gates. Proxy ranking is skipped.')
    top_proxy_dir = top_root / 'top_proxy'
    top_proxy_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = eligible_rows[: cfg.top_n]
    top_proxy_csv = top_proxy_dir / 'top_proxy.csv'
    _write_csv(top_proxy_csv, [{**row, 'rank': idx + 1} for idx, row in enumerate(selected_rows)], ['rank'] + fieldnames)
    for idx, row in enumerate(selected_rows, start=1):
        src = Path(str(row['fm_xml']))
        dst = top_proxy_dir / f'rank{idx:02d}_{src.name}'
        shutil.copy2(src, dst)
    manifest = {
        'pipeline_root': str(pipeline_root),
        'summary_table': str(summary_csv),
        'top_proxy_table': str(top_proxy_csv),
        'total_candidates': len(ranked_rows),
        'eligible_candidates': len(eligible_rows),
        'top_n': cfg.top_n,
        'require_sat': cfg.require_sat,
        'use_evidence': cfg.use_evidence,
        'use_consensus': cfg.use_consensus,
    }
    (top_root / 'proxy_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
