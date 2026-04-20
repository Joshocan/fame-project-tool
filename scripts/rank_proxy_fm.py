#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def _collect_candidates(fm_dir: Path, patterns: List[str]) -> List[dict]:
    seen = set()
    candidates = []
    for pattern in patterns:
        for path in sorted(fm_dir.glob(pattern)):
            resolved = path.expanduser().resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            candidates.append({'run_id': resolved.stem, 'fm_xml': str(resolved), 'meta': ''})
    return candidates


def main() -> None:
    print('[rank_proxy_fm] Starting proxy ranker...')
    print('[rank_proxy_fm] Loading configuration and evaluation modules...')

    from fame.config.load import load_config
    from fame.evaluation.proxy_consensus import ProxyConsensusConfig
    from fame.evaluation.proxy_evidence import ProxyEvidenceConfig
    from fame.evaluation.proxy_selector import ProxySelectorConfig, rank_proxy_fms
    from fame.utils.dirs import build_paths

    cfg_yaml = load_config()
    paths = build_paths()
    default_xsd = paths.specifications / 'feature_model_featureide.xsd'

    ap = argparse.ArgumentParser(description='Rank feature models using the proxy selector (no ground truth required).')
    ap.add_argument('--fm-dir', required=True, help='Directory containing candidate FM XML files')
    ap.add_argument('--chunks-dir', default=str(Path('data/processed/algorithm_1/chunks')), help='Directory containing *.chunks.json files')
    ap.add_argument('--outdir', default='', help='Output root directory; defaults to --fm-dir parent')
    ap.add_argument('--xsd-path', default=str(default_xsd), help='FeatureIDE XSD path')
    ap.add_argument('--top-fm', type=int, default=cfg_yaml.outputs.top_fm, help='Top admissible FMs to copy')
    ap.add_argument('--glob', action='append', default=[], help='Glob pattern(s) inside --fm-dir')
    ap.add_argument('--evidence-model-name', default=cfg_yaml.evaluation.coverage.model_name, help='SentenceTransformer model for evidence scoring')
    ap.add_argument('--support-threshold', type=float, default=cfg_yaml.evaluation.coverage.similarity_threshold, help='Evidence support threshold')
    ap.add_argument('--max-chunks', type=int, default=0, help='Optional cap on number of chunks loaded for evidence scoring (0 = all)')
    ap.add_argument('--require-sat', action='store_true', help='Require SAT satisfiability for admissibility')
    args = ap.parse_args()

    fm_dir = Path(args.fm_dir).expanduser().resolve()
    if not fm_dir.exists():
        raise FileNotFoundError(f'FM directory not found: {fm_dir}')
    chunks_dir = Path(args.chunks_dir).expanduser().resolve()
    if not chunks_dir.exists():
        raise FileNotFoundError(f'Chunks directory not found: {chunks_dir}')

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else fm_dir.parent
    xsd_path = Path(args.xsd_path).expanduser().resolve() if args.xsd_path else None
    patterns = list(args.glob) if args.glob else ['*.xml']
    candidates = _collect_candidates(fm_dir, patterns)
    if not candidates:
        raise FileNotFoundError(f'No FM XML files matched in {fm_dir} for patterns: {patterns}')

    print(f'[rank_proxy_fm] Pipeline root: {outdir}')
    print(f'[rank_proxy_fm] FM dir       : {fm_dir}')
    print(f'[rank_proxy_fm] Chunks dir   : {chunks_dir}')
    print(f'[rank_proxy_fm] Candidates   : {len(candidates)}')
    print(f'[rank_proxy_fm] Patterns     : {patterns}')
    print(f'[rank_proxy_fm] Top-N        : {args.top_fm}')
    print(f'[rank_proxy_fm] Require SAT  : {bool(args.require_sat)}')

    manifest = rank_proxy_fms(
        candidates=candidates,
        pipeline_root=outdir,
        cfg=ProxySelectorConfig(
            top_n=max(0, int(args.top_fm)),
            chunks_dir=chunks_dir,
            xsd_path=xsd_path,
            evidence=ProxyEvidenceConfig(
                model_name=str(args.evidence_model_name),
                support_threshold=float(args.support_threshold),
                max_chunks=max(0, int(args.max_chunks)),
            ),
            consensus=ProxyConsensusConfig(),
            output_subdir=f'top_{max(0, int(args.top_fm))}',
            require_sat=bool(args.require_sat),
        ),
    )
    if not manifest:
        print('No proxy ranking output produced.')
        return
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
