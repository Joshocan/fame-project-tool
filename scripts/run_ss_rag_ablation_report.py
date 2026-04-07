#!/usr/bin/env python3
"""One-click SS-RAG ablation reporting: build dataset then generate tables/plots."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_ss_k_rag_ablation_data.py"
PLOT_SCRIPT = REPO_ROOT / "scripts" / "plot_ss_rag_ablation.py"


def _run(cmd: List[str], *, env: dict) -> None:
    print(f"→ {' '.join(shlex.quote(x) for x in cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run SS-RAG ablation post-processing end-to-end (build dataset + plot report)."
    )
    ap.add_argument(
        "--manifests",
        nargs="+",
        default=["results/rag/ss-rgfm/ablation/ss_rgfm_k_runs_*.json"],
        help="Manifest files/directories/globs passed to build_ss_rag_ablation_data.py",
    )
    ap.add_argument("--gt", default="", help="Optional ground-truth XML path for coverage scoring")
    ap.add_argument(
        "--xsd",
        default="prompts/specifications/feature_model_featureide.xsd",
        help="XSD path for validation during dataset build",
    )
    ap.add_argument("--coverage-model", default="")
    ap.add_argument("--coverage-threshold", type=float, default=None)
    ap.add_argument("--coverage-top-k", type=int, default=None)
    ap.add_argument("--coverage-feature-weight", type=float, default=None)
    ap.add_argument("--coverage-parent-weight", type=float, default=None)
    ap.add_argument("--require-coverage", action="store_true")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument(
        "--analysis-out-dir",
        default="results/rag/ss-rgfm/analysis",
        help="Parent output dir for built datasets",
    )
    ap.add_argument("--report-out-dir", default="", help="Optional explicit report output directory")
    ap.add_argument("--label", default="", help="Dataset label (default: generated timestamp label)")
    ap.add_argument(
        "--python",
        default="",
        help="Python executable to use (default: current interpreter)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    py = Path(args.python).expanduser().resolve() if args.python.strip() else Path(sys.executable).resolve()
    if not py.exists():
        raise FileNotFoundError(f"Python executable not found: {py}")

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    label = args.label.strip() or f"ablation_report_{ts}"

    analysis_out_dir = Path(args.analysis_out_dir).expanduser().resolve()
    data_dir = analysis_out_dir / label
    report_out_dir = Path(args.report_out_dir).expanduser().resolve() if args.report_out_dir.strip() else (data_dir / "report")

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    env.setdefault("FAME_BASE_DIR", str(REPO_ROOT))

    build_cmd = [
        str(py),
        str(BUILD_SCRIPT),
        "--manifests",
        *args.manifests,
        "--xsd",
        args.xsd,
        "--out-dir",
        str(analysis_out_dir),
        "--label",
        label,
    ]
    if args.gt.strip():
        build_cmd += ["--gt", args.gt]
    if args.coverage_model.strip():
        build_cmd += ["--coverage-model", args.coverage_model]
    if args.coverage_threshold is not None:
        build_cmd += ["--coverage-threshold", str(args.coverage_threshold)]
    if args.coverage_top_k is not None:
        build_cmd += ["--coverage-top-k", str(args.coverage_top_k)]
    if args.coverage_feature_weight is not None:
        build_cmd += ["--coverage-feature-weight", str(args.coverage_feature_weight)]
    if args.coverage_parent_weight is not None:
        build_cmd += ["--coverage-parent-weight", str(args.coverage_parent_weight)]
    if args.require_coverage:
        build_cmd += ["--require-coverage"]
    if args.include_failed:
        build_cmd += ["--include-failed"]

    plot_cmd = [
        str(py),
        str(PLOT_SCRIPT),
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(report_out_dir),
    ]

    print("==================== SS-RAG Ablation Report Runner ====================", flush=True)
    print(f"Python          : {py}", flush=True)
    print(f"Repo            : {REPO_ROOT}", flush=True)
    print(f"Dataset label   : {label}", flush=True)
    print(f"Data dir        : {data_dir}", flush=True)
    print(f"Report dir      : {report_out_dir}", flush=True)
    print("======================================================================", flush=True)

    print("Step 1/2: Building enriched ablation dataset...", flush=True)
    _run(build_cmd, env=env)
    print("Step 2/2: Generating tables and plots...", flush=True)
    _run(plot_cmd, env=env)

    print("\nSUCCESS: End-to-end ablation report completed", flush=True)
    print(f"Data dir   : {data_dir}", flush=True)
    print(f"Report dir : {report_out_dir}", flush=True)


if __name__ == "__main__":
    main()
