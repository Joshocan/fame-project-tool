#!/usr/bin/env python3
"""Build an overall pipeline-performance dataset across the four core pipelines."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from fame.config.load import load_config
from fame.evaluation import (
    CoverageConfig,
    CoverageEvaluator,
    analyze_sat_quality,
    edge_jaccard_vs_gt,
    extract_feature_list,
    feature_diff_stats,
    parent_match_rate,
    semantic_prf,
    validate_feature_model,
)
from fame.utils.dirs import ensure_dir


PIPELINE_SPECS: Dict[str, Dict[str, str]] = {
    "ss_rag": {
        "reports_dir": "results/rag/ss-rgfm/reports",
        "fm_dir": "results/rag/ss-rgfm/fm",
        "xml_suffix": ".xml",
    },
    "is_rag": {
        "reports_dir": "results/rag/is-rgfm/reports",
        "fm_dir": "results/rag/is-rgfm/fm",
        "xml_suffix": ".final.xml",
    },
    "ss_nonrag": {
        "reports_dir": "results/non_rag/ss-nonrag/reports",
        "fm_dir": "results/non_rag/ss-nonrag/fm",
        "xml_suffix": ".xml",
    },
    "is_nonrag": {
        "reports_dir": "results/non_rag/is-nonrag/reports",
        "fm_dir": "results/non_rag/is-nonrag/fm",
        "xml_suffix": ".final.xml",
    },
}


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    try:
        return float(v)
    except Exception:
        return None


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    try:
        return int(v)
    except Exception:
        return None


def _mean(vals: Sequence[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def _safe_rate(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return round(num / den, 4)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})


def _discover_meta_files(
    *,
    repo_root: Path,
    pipelines: Sequence[str],
    include_heqed: bool,
) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for pipeline in pipelines:
        spec = PIPELINE_SPECS[pipeline]
        reports_dir = repo_root / spec["reports_dir"]
        files = sorted(reports_dir.glob("*.meta.json"))
        picked: List[Path] = []
        for f in files:
            name = f.name.lower()
            if "response" not in name:
                continue
            if not include_heqed and "heqed" in name:
                continue
            picked.append(f.resolve())
        out[pipeline] = picked
    return out


def _resolve_xml_path(
    *,
    repo_root: Path,
    pipeline: str,
    meta_path: Path,
    meta: Dict[str, Any],
) -> Optional[Path]:
    for key in ("final_xml", "fm_xml", "xml"):
        raw = meta.get(key)
        if not raw:
            continue
        p = Path(str(raw)).expanduser()
        resolved = p.resolve() if p.is_absolute() else (repo_root / p).resolve()
        if resolved.exists():
            return resolved

    run_id = str(meta.get("run_id") or "").strip()
    if not run_id:
        return None

    spec = PIPELINE_SPECS[pipeline]
    candidate = (repo_root / spec["fm_dir"] / f"{run_id}{spec['xml_suffix']}").resolve()
    if candidate.exists():
        return candidate
    return None


def _extract_runtime_seconds(meta: Dict[str, Any]) -> Optional[float]:
    return _as_float(meta.get("total_llm_duration_seconds") or meta.get("llm_duration_seconds"))


def _extract_collection_mode(pipeline: str, meta: Dict[str, Any]) -> str:
    mode = str(meta.get("collection_mode") or "").strip()
    if mode:
        return mode
    if pipeline == "ss_rag":
        return "unknown"
    if pipeline == "is_rag":
        return "iterative_per_source"
    return "context_conditioned"


def _init_coverage(gt_path: Optional[Path], args: argparse.Namespace) -> Tuple[Optional[CoverageEvaluator], Optional[Path]]:
    if gt_path is None:
        return None, None
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth XML not found: {gt_path}")

    cfg_doc = load_config()
    cov_cfg = cfg_doc.evaluation.coverage
    cfg = CoverageConfig(
        model_name=args.coverage_model or cov_cfg.model_name,
        similarity_threshold=args.coverage_threshold if args.coverage_threshold is not None else cov_cfg.similarity_threshold,
        top_k=args.coverage_top_k if args.coverage_top_k is not None else cov_cfg.top_k,
        feature_weight=args.coverage_feature_weight if args.coverage_feature_weight is not None else cov_cfg.feature_weight,
        parent_weight=args.coverage_parent_weight if args.coverage_parent_weight is not None else cov_cfg.parent_weight,
    )
    try:
        return CoverageEvaluator(cfg), gt_path
    except Exception as e:
        if args.require_coverage:
            raise RuntimeError(f"Failed to initialize coverage evaluator: {e}") from e
        print(f"WARN: Coverage evaluator unavailable, continuing without semantic metrics. Details: {e}")
        return None, gt_path


def _extract_run_rows(
    *,
    repo_root: Path,
    pipeline: str,
    meta_paths: Sequence[Path],
    coverage_eval: Optional[CoverageEvaluator],
    gt_path: Optional[Path],
    xsd_path: Optional[Path],
    require_sat: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for idx, meta_path in enumerate(meta_paths, start=1):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        xml_path = _resolve_xml_path(repo_root=repo_root, pipeline=pipeline, meta_path=meta_path, meta=meta)

        base_row: Dict[str, Any] = {
            "pipeline": pipeline,
            "meta_file": str(meta_path),
            "run_id": meta.get("run_id"),
            "root_feature": meta.get("root_feature"),
            "domain": meta.get("domain"),
            "model": meta.get("llm_model"),
            "llm_host": meta.get("llm_host"),
            "collection_mode": _extract_collection_mode(pipeline, meta),
            "num_sources": _as_int(meta.get("num_sources")),
            "num_chunks_total": _as_int(meta.get("num_chunks_total")),
            "num_evidence_chunks": _as_int(meta.get("num_evidence_chunks")),
            "context_chars": _as_int(meta.get("context_chars")),
            "llm_duration_seconds": _extract_runtime_seconds(meta),
            "fm_xml": str(xml_path) if xml_path else None,
            "feature_count": None,
            "abstract_feature_count": None,
            "concrete_feature_count": None,
            "wellformed_ok": None,
            "wellformed_error_count": None,
            "wellformed_errors": None,
            "satisfiable": None,
            "dead_features_count": None,
            "core_features_count": None,
            "sat_error": None,
            "coverage_score": None,
            "coverage_error": None,
            "semantic_precision": None,
            "semantic_recall": None,
            "semantic_f1": None,
            "missing_feature_count": None,
            "extra_feature_count": None,
            "missing_feature_ratio": None,
            "extra_feature_ratio": None,
            "edge_jaccard_to_gt": None,
            "parent_match_rate": None,
            "feature_extract_error": None,
        }

        if xml_path is None:
            rows.append(base_row)
            continue

        wf = validate_feature_model(xml_path, xsd_path)
        base_row["wellformed_ok"] = wf.ok
        base_row["wellformed_error_count"] = len(wf.errors)
        base_row["wellformed_errors"] = wf.errors

        try:
            feats = extract_feature_list(xml_path)
            base_row["feature_count"] = len(feats)
            base_row["abstract_feature_count"] = sum(1 for f in feats if f.feature_type == "abstract")
            base_row["concrete_feature_count"] = sum(1 for f in feats if f.feature_type == "concrete")
        except Exception as e:
            base_row["feature_extract_error"] = str(e)

        if wf.ok:
            try:
                sat = analyze_sat_quality(xml_path, compute_products=False)
                base_row["satisfiable"] = sat.satisfiable
                base_row["dead_features_count"] = len(sat.dead_features or [])
                base_row["core_features_count"] = len(sat.core_features or [])
            except Exception as e:
                base_row["sat_error"] = str(e)
                if require_sat:
                    raise
        else:
            base_row["sat_error"] = "Skipped SAT: XML not well-formed/XSD-valid"

        if coverage_eval is not None and gt_path is not None:
            try:
                base_row["coverage_score"] = coverage_eval.score(gt_path, xml_path, verbose=False)
            except Exception as e:
                base_row["coverage_error"] = str(e)

            try:
                prf = semantic_prf(
                    gt_path,
                    xml_path,
                    model=coverage_eval.model,
                    threshold=coverage_eval.cfg.similarity_threshold,
                )
                base_row.update(prf)
                base_row.update(feature_diff_stats(gt_path, xml_path))
                base_row["edge_jaccard_to_gt"] = edge_jaccard_vs_gt(gt_path, xml_path)
                base_row["parent_match_rate"] = parent_match_rate(gt_path, xml_path)
            except Exception as e:
                base_row["coverage_error"] = str(e)

        rows.append(base_row)

        if idx == 1 or idx == len(meta_paths) or idx % 25 == 0:
            print(f"[{pipeline}] processed runs: {idx}/{len(meta_paths)}", flush=True)

    return rows


def _summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("pipeline") or "unknown"),
            str(row.get("model") or "unknown"),
            str(row.get("domain") or "unknown"),
            str(row.get("root_feature") or "unknown"),
        )
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(grouped):
        pipeline, model, domain, root_feature = key
        grp = grouped[key]
        wf_vals = [r.get("wellformed_ok") for r in grp if r.get("wellformed_ok") is not None]
        sat_vals = [r.get("satisfiable") for r in grp if r.get("satisfiable") is not None]
        row = {
            "pipeline": pipeline,
            "model": model,
            "domain": domain,
            "root_feature": root_feature,
            "runs_total": len(grp),
            "runs_with_xml": sum(1 for r in grp if r.get("fm_xml")),
            "wellformed_rate": _safe_rate(sum(1 for v in wf_vals if v), len(wf_vals)) if wf_vals else None,
            "satisfiable_rate": _safe_rate(sum(1 for v in sat_vals if v), len(sat_vals)) if sat_vals else None,
            "coverage_score_mean": _mean([_as_float(r.get("coverage_score")) for r in grp]),
            "semantic_precision_mean": _mean([_as_float(r.get("semantic_precision")) for r in grp]),
            "semantic_recall_mean": _mean([_as_float(r.get("semantic_recall")) for r in grp]),
            "semantic_f1_mean": _mean([_as_float(r.get("semantic_f1")) for r in grp]),
            "llm_duration_seconds_mean": _mean([_as_float(r.get("llm_duration_seconds")) for r in grp]),
            "feature_count_mean": _mean([_as_float(r.get("feature_count")) for r in grp]),
            "abstract_feature_count_mean": _mean([_as_float(r.get("abstract_feature_count")) for r in grp]),
            "concrete_feature_count_mean": _mean([_as_float(r.get("concrete_feature_count")) for r in grp]),
            "dead_features_count_mean": _mean([_as_float(r.get("dead_features_count")) for r in grp]),
            "core_features_count_mean": _mean([_as_float(r.get("core_features_count")) for r in grp]),
            "edge_jaccard_to_gt_mean": _mean([_as_float(r.get("edge_jaccard_to_gt")) for r in grp]),
            "parent_match_rate_mean": _mean([_as_float(r.get("parent_match_rate")) for r in grp]),
        }
        out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build an overall pipeline-performance dataset.")
    ap.add_argument(
        "--pipelines",
        nargs="+",
        choices=sorted(PIPELINE_SPECS.keys()),
        default=sorted(PIPELINE_SPECS.keys()),
        help="Which pipelines to include.",
    )
    ap.add_argument("--gt", default="", help="Ground-truth XML for semantic coverage / semantic F1.")
    ap.add_argument(
        "--xsd",
        default="prompts/specifications/feature_model_featureide.xsd",
        help="XSD path for well-formedness validation.",
    )
    ap.add_argument("--coverage-model", default="")
    ap.add_argument("--coverage-threshold", type=float, default=None)
    ap.add_argument("--coverage-top-k", type=int, default=None)
    ap.add_argument("--coverage-feature-weight", type=float, default=None)
    ap.add_argument("--coverage-parent-weight", type=float, default=None)
    ap.add_argument("--require-coverage", action="store_true")
    ap.add_argument("--require-sat", action="store_true", help="Fail if SAT analysis is unavailable.")
    ap.add_argument("--include-heqed", action="store_true", help="Include HEQED runs in addition to generic response runs.")
    ap.add_argument("--out-dir", default="results/analysis", help="Output directory root.")
    ap.add_argument("--label", default="", help="Optional output folder label.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()

    gt_path = Path(args.gt).expanduser().resolve() if args.gt.strip() else None
    xsd_path = Path(args.xsd).expanduser().resolve() if args.xsd.strip() else None
    if xsd_path is not None and not xsd_path.exists():
        xsd_path = None

    coverage_eval, gt_path = _init_coverage(gt_path, args)

    discovered = _discover_meta_files(
        repo_root=repo_root,
        pipelines=args.pipelines,
        include_heqed=args.include_heqed,
    )
    total_meta = sum(len(v) for v in discovered.values())
    if total_meta == 0:
        raise FileNotFoundError("No matching .meta.json files found for the selected pipelines.")

    print(f"Discovered run metadata files: {total_meta}", flush=True)
    for pipeline, files in discovered.items():
        print(f"  - {pipeline}: {len(files)}", flush=True)

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    label = args.label.strip() or f"overall_pipeline_data_{ts}"
    out_dir = ensure_dir((repo_root / args.out_dir).resolve() / label)
    print(f"Output directory: {out_dir}", flush=True)

    all_rows: List[Dict[str, Any]] = []
    for pipeline in args.pipelines:
        meta_paths = discovered[pipeline]
        all_rows.extend(
            _extract_run_rows(
                repo_root=repo_root,
                pipeline=pipeline,
                meta_paths=meta_paths,
                coverage_eval=coverage_eval,
                gt_path=gt_path,
                xsd_path=xsd_path,
                require_sat=args.require_sat,
            )
        )

    summary_rows = _summarize(all_rows)
    wf_rows = [r for r in all_rows if r.get("wellformed_ok") is True]
    wf_summary_rows = _summarize(wf_rows)

    runs_json = out_dir / "overall_pipeline_runs_enriched.json"
    runs_csv = out_dir / "overall_pipeline_runs_enriched.csv"
    summary_json = out_dir / "overall_pipeline_summary.json"
    summary_csv = out_dir / "overall_pipeline_summary.csv"
    summary_wf_json = out_dir / "overall_pipeline_summary_wf_only.json"
    summary_wf_csv = out_dir / "overall_pipeline_summary_wf_only.csv"
    meta_json = out_dir / "dataset_meta.json"

    runs_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    summary_wf_json.write_text(json.dumps(wf_summary_rows, indent=2), encoding="utf-8")

    run_fields = [
        "pipeline",
        "run_id",
        "model",
        "domain",
        "root_feature",
        "collection_mode",
        "llm_host",
        "meta_file",
        "fm_xml",
        "num_sources",
        "num_chunks_total",
        "num_evidence_chunks",
        "context_chars",
        "llm_duration_seconds",
        "wellformed_ok",
        "wellformed_error_count",
        "satisfiable",
        "dead_features_count",
        "core_features_count",
        "coverage_score",
        "semantic_precision",
        "semantic_recall",
        "semantic_f1",
        "feature_count",
        "abstract_feature_count",
        "concrete_feature_count",
        "missing_feature_count",
        "extra_feature_count",
        "missing_feature_ratio",
        "extra_feature_ratio",
        "edge_jaccard_to_gt",
        "parent_match_rate",
        "sat_error",
        "coverage_error",
        "feature_extract_error",
    ]
    _write_csv(runs_csv, all_rows, run_fields)

    summary_fields = [
        "pipeline",
        "model",
        "domain",
        "root_feature",
        "runs_total",
        "runs_with_xml",
        "wellformed_rate",
        "satisfiable_rate",
        "coverage_score_mean",
        "semantic_precision_mean",
        "semantic_recall_mean",
        "semantic_f1_mean",
        "llm_duration_seconds_mean",
        "feature_count_mean",
        "abstract_feature_count_mean",
        "concrete_feature_count_mean",
        "dead_features_count_mean",
        "core_features_count_mean",
        "edge_jaccard_to_gt_mean",
        "parent_match_rate_mean",
    ]
    _write_csv(summary_csv, summary_rows, summary_fields)
    _write_csv(summary_wf_csv, wf_summary_rows, summary_fields)

    meta = {
        "created_at": ts,
        "label": label,
        "pipelines": list(args.pipelines),
        "include_heqed": bool(args.include_heqed),
        "ground_truth_xml": str(gt_path) if gt_path else None,
        "xsd": str(xsd_path) if xsd_path else None,
        "run_rows": len(all_rows),
        "wellformed_run_rows": len(wf_rows),
        "summary_rows": len(summary_rows),
        "summary_rows_wellformed_only": len(wf_summary_rows),
        "meta_files_by_pipeline": {k: [str(p) for p in v] for k, v in discovered.items()},
        "files": {
            "runs_json": str(runs_json),
            "runs_csv": str(runs_csv),
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "summary_wf_json": str(summary_wf_json),
            "summary_wf_csv": str(summary_wf_csv),
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("SUCCESS: Built overall pipeline dataset")
    print(f"Output dir   : {out_dir}")
    print(f"Run rows     : {len(all_rows)}")
    print(f"Summary rows : {len(summary_rows)}")
    print(f"Runs CSV     : {runs_csv}")
    print(f"Summary CSV  : {summary_csv}")


if __name__ == "__main__":
    main()
