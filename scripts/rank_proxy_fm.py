#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_local_module(name: str):
    module_path = ROOT / 'fame' / 'evaluation' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'paper2_{name}', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module: {module_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_proxy_selector = _load_local_module('proxy_selector')
ProxyRankConfig = _proxy_selector.ProxyRankConfig
rank_proxy_fms = _proxy_selector.rank_proxy_fms


def _collect_candidates(fm_dir: Path, patterns: list[str]) -> list[dict]:
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
    ap = argparse.ArgumentParser(description='Rank candidate FMs using proxy-based trustworthy selection.')
    ap.add_argument('--fm-dir', required=True)
    ap.add_argument('--chunks-dir', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--xsd-path', default='')
    ap.add_argument('--top-fm', type=int, default=5)
    ap.add_argument('--glob', action='append', default=[])
    ap.add_argument('--require-sat', action='store_true')
    ap.add_argument('--selector-use-evidence', default='true')
    ap.add_argument('--selector-use-consensus', default='true')
    ap.add_argument('--evidence-weight', type=float, default=0.7)
    ap.add_argument('--consensus-weight', type=float, default=0.3)
    args = ap.parse_args()

    fm_dir = Path(args.fm_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    patterns = list(args.glob) if args.glob else ['*.xml']
    candidates = _collect_candidates(fm_dir, patterns)
    if not candidates:
        raise FileNotFoundError(f'No FM XML files matched in {fm_dir} for patterns: {patterns}')
    manifest = rank_proxy_fms(
        candidates=candidates,
        pipeline_root=outdir,
        cfg=ProxyRankConfig(
            top_n=max(0, int(args.top_fm)),
            chunks_dir=Path(args.chunks_dir).expanduser().resolve(),
            xsd_path=Path(args.xsd_path).expanduser().resolve() if args.xsd_path else None,
            output_subdir=f'top_{max(0, int(args.top_fm))}',
            require_sat=bool(args.require_sat),
            use_evidence=str(args.selector_use_evidence).lower() == 'true',
            use_consensus=str(args.selector_use_consensus).lower() == 'true',
            evidence_weight=float(args.evidence_weight),
            consensus_weight=float(args.consensus_weight),
        ),
    )
    if manifest:
        print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
