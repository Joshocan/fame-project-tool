from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .constraints import extract_constraints
from .feature_list import extract_feature_list
from .proxy_consensus import (
    CandidateSignature,
    ProxyConsensusConfig,
    build_candidate_signature,
    score_consensus,
)
from .proxy_evidence import ProxyEvidenceConfig, prepare_evidence_context, score_evidence
from .quality_sat import analyze_sat_quality
from .wellformed import validate_feature_model


@dataclass(frozen=True)
class ProxySelectorConfig:
    top_n: int
    chunks_dir: Path
    xsd_path: Optional[Path]
    evidence: ProxyEvidenceConfig
    consensus: ProxyConsensusConfig
    output_subdir: str = ''
    require_sat: bool = True
    evidence_weight: float = 0.7
    consensus_weight: float = 0.3


def _duplicate_feature_names(xml_path: Path) -> List[str]:
    counts = Counter(rec.feature_name for rec in extract_feature_list(xml_path))
    return sorted(name for name, count in counts.items() if count > 1)


def _constraints_valid(xml_path: Path) -> tuple[bool, str]:
    xml_text = xml_path.read_text(encoding='utf-8', errors='ignore')
    if '<constraints' not in xml_text:
        return True, ''
    try:
        extract_constraints(xml_path)
        return True, ''
    except Exception as exc:
        return False, str(exc)


def _summary_sort_key(row: Dict[str, Any]) -> tuple:
    return (
        1 if row.get('eligible_ok') else 0,
        float(row.get('proxy_score') or -1),
        float(row.get('evidence_score') or -1),
        float(row.get('consensus_score') or -1),
        -int(row.get('candidate_index') or 0),
    )


def _safe_name(name: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '-' for ch in name)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def _proxy_score(evidence_score: Optional[float], consensus_score: Optional[float], cfg: ProxySelectorConfig) -> Optional[float]:
    if evidence_score is None or consensus_score is None:
        return None
    return round(float(cfg.evidence_weight) * float(evidence_score) + float(cfg.consensus_weight) * float(consensus_score), 6)


def build_proxy_ranked_rows(*, candidates: Sequence[Dict[str, Any]], cfg: ProxySelectorConfig) -> List[Dict[str, Any]]:
    ranked_rows: List[Dict[str, Any]] = []
    signatures: Dict[str, CandidateSignature] = {}
    print(f'[proxy_selector] Evaluating {len(candidates)} candidate FM files...')
    print(f'[proxy_selector] Preparing evidence context from chunks: {cfg.chunks_dir}')
    evidence_context = prepare_evidence_context(cfg.chunks_dir, cfg.evidence)
    print(f'[proxy_selector] Evidence context ready: {len(evidence_context.chunks)} chunks cached, model loaded once.')

    for idx, candidate in enumerate(candidates, start=1):
        fm_xml = Path(str(candidate.get('fm_xml') or candidate.get('final_xml') or '')).expanduser().resolve()
        run_id = str(candidate.get('run_id') or fm_xml.stem)
        print(f'[proxy_selector] [{idx}/{len(candidates)}] Checking {run_id}')
        if not fm_xml.exists():
            print(f'[proxy_selector] [{idx}/{len(candidates)}] Skipping missing file: {fm_xml}')
            continue

        wf = validate_feature_model(fm_xml, xsd_path=cfg.xsd_path)
        duplicate_feature_names: List[str] = []
        duplicate_feature_error = ''
        try:
            duplicate_feature_names = _duplicate_feature_names(fm_xml)
        except Exception as exc:
            duplicate_feature_error = str(exc)
        has_duplicate_features = len(duplicate_feature_names) > 0

        constraints_valid, constraints_error = (True, '')
        if bool(wf.ok) and not has_duplicate_features and not duplicate_feature_error:
            constraints_valid, constraints_error = _constraints_valid(fm_xml)

        satisfiable: Optional[bool] = None
        sat_error = ''
        if bool(wf.ok) and not has_duplicate_features and not duplicate_feature_error:
            try:
                sat = analyze_sat_quality(fm_xml, compute_products=False)
                satisfiable = sat.satisfiable
            except Exception as exc:
                sat_error = str(exc)

        eligible_ok = bool(wf.ok) and not has_duplicate_features and not duplicate_feature_error
        if cfg.require_sat:
            eligible_ok = eligible_ok and bool(satisfiable) and not sat_error

        evidence_score = None
        trace_present_rate = None
        label_support_mean = None
        trace_support_mean = None
        supported_feature_rate = None
        consensus_score = None
        feature_overlap_mean = None
        edge_overlap_mean = None
        constraint_overlap_mean = None

        if eligible_ok:
            evidence = score_evidence(fm_xml, cfg.chunks_dir, cfg.evidence, context=evidence_context)
            evidence_score = evidence.evidence_score
            trace_present_rate = evidence.trace_present_rate
            label_support_mean = evidence.label_support_mean
            trace_support_mean = evidence.trace_support_mean
            supported_feature_rate = evidence.supported_feature_rate
            signatures[str(fm_xml)] = build_candidate_signature(fm_xml)
        else:
            reasons: List[str] = []
            if not wf.ok:
                reasons.append(f'xsd_or_wellformed_errors={len(wf.errors)}')
            if has_duplicate_features:
                reasons.append(f'duplicate_features={len(duplicate_feature_names)}')
            if duplicate_feature_error:
                reasons.append('duplicate_check_failed')
            if not constraints_valid and constraints_error:
                reasons.append('constraints_invalid')
            if cfg.require_sat and sat_error:
                reasons.append('sat_check_failed')
            if cfg.require_sat and satisfiable is False:
                reasons.append('unsatisfiable')
            print(f"[proxy_selector] [{idx}/{len(candidates)}] Ineligible: {', '.join(reasons) if reasons else 'unknown reason'}")

        ranked_rows.append(
            {
                'candidate_index': idx,
                'run_id': candidate.get('run_id') or run_id,
                'fm_xml': str(fm_xml),
                'meta': candidate.get('meta'),
                'wellformed_ok': wf.ok,
                'wellformed_error_count': len(wf.errors),
                'has_duplicate_features': has_duplicate_features,
                'duplicate_feature_count': len(duplicate_feature_names),
                'duplicate_feature_names': duplicate_feature_names,
                'duplicate_feature_error': duplicate_feature_error,
                'constraints_valid': constraints_valid,
                'constraints_error': constraints_error,
                'satisfiable': satisfiable,
                'sat_error': sat_error,
                'eligible_ok': eligible_ok,
                'trace_present_rate': trace_present_rate,
                'label_support_mean': label_support_mean,
                'trace_support_mean': trace_support_mean,
                'supported_feature_rate': supported_feature_rate,
                'evidence_score': evidence_score,
                'feature_overlap_mean': feature_overlap_mean,
                'edge_overlap_mean': edge_overlap_mean,
                'constraint_overlap_mean': constraint_overlap_mean,
                'consensus_score': consensus_score,
                'proxy_score': None,
            }
        )

    total_rows = len(ranked_rows)
    parse_ok_count = sum(1 for r in ranked_rows if r.get('wellformed_ok'))
    duplicate_free_count = sum(1 for r in ranked_rows if not r.get('has_duplicate_features'))
    constraints_valid_count = sum(1 for r in ranked_rows if r.get('constraints_valid'))
    sat_ok_count = sum(1 for r in ranked_rows if r.get('satisfiable') is True)
    eligible_rows = [r for r in ranked_rows if r.get('eligible_ok')]

    print('[proxy_selector] Gate summary:')
    print(f'[proxy_selector]   Parsed/XSD-valid      : {parse_ok_count}/{total_rows}')
    print(f'[proxy_selector]   Duplicate-free        : {duplicate_free_count}/{total_rows}')
    print(f'[proxy_selector]   Constraints-valid     : {constraints_valid_count}/{total_rows}')
    print(f'[proxy_selector]   Satisfiable           : {sat_ok_count}/{total_rows}')
    print(f'[proxy_selector]   Passed all hard gates : {len(eligible_rows)}/{total_rows}')

    if not eligible_rows:
        print('[proxy_selector] No candidates passed the hard gates. Proxy ranking is skipped.')
        return ranked_rows

    for row in eligible_rows:
        sig = signatures.get(str(row['fm_xml']))
        peers = [signatures[str(other['fm_xml'])] for other in eligible_rows if str(other['fm_xml']) != str(row['fm_xml'])]
        consensus = score_consensus(sig, peers, cfg.consensus) if sig else None
        if consensus:
            row['feature_overlap_mean'] = consensus.feature_overlap_mean
            row['edge_overlap_mean'] = consensus.edge_overlap_mean
            row['constraint_overlap_mean'] = consensus.constraint_overlap_mean
            row['consensus_score'] = consensus.consensus_score
            row['proxy_score'] = _proxy_score(row.get('evidence_score'), row.get('consensus_score'), cfg)
    return ranked_rows


def rank_proxy_fms(*, candidates: Sequence[Dict[str, Any]], pipeline_root: Path, cfg: ProxySelectorConfig) -> Optional[Dict[str, Any]]:
    if cfg.top_n <= 0 or not candidates:
        return None

    pipeline_root = Path(pipeline_root)
    print(f'[proxy_selector] Ranking into: {pipeline_root}')
    ranked_rows = build_proxy_ranked_rows(candidates=candidates, cfg=cfg)
    summary_rows = sorted(ranked_rows, key=_summary_sort_key, reverse=True)

    proxy_root = pipeline_root / 'proxy_fm'
    if cfg.output_subdir:
        proxy_root = proxy_root / cfg.output_subdir
    proxy_root.mkdir(parents=True, exist_ok=True)

    _write_csv(
        proxy_root / 'proxy_scores.csv',
        summary_rows,
        [
            'candidate_index', 'run_id', 'fm_xml', 'meta', 'wellformed_ok', 'wellformed_error_count',
            'has_duplicate_features', 'duplicate_feature_count', 'duplicate_feature_names', 'duplicate_feature_error',
            'constraints_valid', 'constraints_error', 'satisfiable', 'sat_error', 'eligible_ok',
            'trace_present_rate', 'label_support_mean', 'trace_support_mean', 'supported_feature_rate',
            'evidence_score', 'feature_overlap_mean', 'edge_overlap_mean', 'constraint_overlap_mean',
            'consensus_score', 'proxy_score',
        ],
    )

    eligible = [r for r in summary_rows if r.get('eligible_ok')]
    print(f"[proxy_selector] Total candidates passing hard gates: {len(eligible)}/{len(summary_rows)}")
    top_proxy_dir = proxy_root / 'top_proxy'
    top_proxy_dir.mkdir(parents=True, exist_ok=True)
    copied_rows: List[Dict[str, Any]] = []
    if eligible:
        selected = eligible[: cfg.top_n]
        for rank, row in enumerate(selected, start=1):
            src = Path(str(row['fm_xml']))
            dst = top_proxy_dir / f'rank{rank:02d}_{_safe_name(src.name)}'
            shutil.copy2(src, dst)
            row_copy = dict(row)
            row_copy['rank'] = rank
            row_copy['copied_xml'] = str(dst)
            copied_rows.append(row_copy)
    else:
        print('[proxy_selector] No admissible candidates. No top proxy files will be copied.')

    _write_csv(
        top_proxy_dir / 'top_proxy.csv',
        copied_rows,
        ['rank', 'run_id', 'fm_xml', 'copied_xml', 'eligible_ok', 'satisfiable', 'trace_present_rate', 'label_support_mean', 'trace_support_mean', 'supported_feature_rate', 'evidence_score', 'feature_overlap_mean', 'edge_overlap_mean', 'constraint_overlap_mean', 'consensus_score', 'proxy_score'],
    )

    manifest: Dict[str, Any] = {
        'pipeline_root': str(pipeline_root),
        'chunks_dir': str(cfg.chunks_dir),
        'top_n': cfg.top_n,
        'require_sat': cfg.require_sat,
        'summary_table': str(proxy_root / 'proxy_scores.csv'),
        'top_proxy_table': str(top_proxy_dir / 'top_proxy.csv'),
        'ranking_rule': 'eligible if parseable, duplicate-free, XSD-valid' + (' and satisfiable' if cfg.require_sat else '') + '; proxy_score = 0.7 * evidence_score + 0.3 * consensus_score',
        'eligible_candidates': len(eligible),
        'total_candidates': len(summary_rows),
    }
    (proxy_root / 'proxy_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
