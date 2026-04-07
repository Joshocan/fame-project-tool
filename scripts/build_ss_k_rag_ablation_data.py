#!/usr/bin/env python3
"""Build enriched SS-RAG k-ablation datasets from run manifests."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

from fame.config.load import load_config
from fame.evaluation import CoverageConfig, CoverageEvaluator, extract_feature_list, validate_feature_model
from fame.evaluation.coverage import extract_nodes, util
from fame.utils.dirs import build_paths, ensure_dir

NODE_TAGS = {"feature", "and", "or", "alt"}
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
UNDERSCORE_RE = re.compile(r"_+")


def _normalize_name(name: str) -> str:
    n = (name or "").strip().lower()
    n = NON_ALNUM_RE.sub("_", n)
    n = UNDERSCORE_RE.sub("_", n).strip("_")
    return n


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


def _resolve_input_paths(inputs: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for raw in inputs:
        if any(ch in raw for ch in ["*", "?", "["]):
            matches = sorted(glob.glob(raw, recursive=True))
            out.extend(Path(m).expanduser().resolve() for m in matches)
            continue
        p = Path(raw).expanduser().resolve()
        if p.is_dir():
            out.extend(sorted(p.glob("ss_rgfm_k_runs_*.json")))
            out.extend(sorted(p.glob("ss_rgfm_k_ablation_*.json")))
        else:
            out.append(p)

    deduped: List[Path] = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return [p for p in deduped if p.exists()]


def _collect_names(xml_path: Path) -> List[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    struct = root.find("struct")
    if struct is None:
        return []

    names: List[str] = []

    def walk(node: ET.Element) -> None:
        for ch in node:
            if ch.tag in NODE_TAGS:
                name = (ch.attrib.get("name") or "").strip()
                if name:
                    names.append(name)
                walk(ch)

    walk(struct)
    return names


def _dup_stats(xml_path: Path) -> Dict[str, Optional[int]]:
    try:
        names = _collect_names(xml_path)
    except Exception:
        return {
            "num_duplications": None,
            "num_duplication_groups": None,
            "num_near_duplications": None,
            "num_near_duplication_groups": None,
        }

    exact = Counter(names)
    norm = Counter(_normalize_name(n) for n in names)
    return {
        "num_duplications": sum(c - 1 for c in exact.values() if c > 1),
        "num_duplication_groups": sum(1 for c in exact.values() if c > 1),
        "num_near_duplications": sum(c - 1 for c in norm.values() if c > 1),
        "num_near_duplication_groups": sum(1 for c in norm.values() if c > 1),
    }


def _semantic_prf(
    human_xml: Path,
    auto_xml: Path,
    *,
    model,
    threshold: float,
) -> Dict[str, Optional[float]]:
    try:
        human_nodes = extract_nodes(human_xml)
        auto_nodes = extract_nodes(auto_xml)
        human_names = [h for h, _ in human_nodes]
        auto_names = [a for a, _ in auto_nodes]
        if not human_names or not auto_names:
            return {"semantic_precision": None, "semantic_recall": None, "semantic_f1": None}
        # Embed
        human_emb = model.encode(human_names, normalize_embeddings=True, convert_to_tensor=True)
        auto_emb = model.encode(auto_names, normalize_embeddings=True, convert_to_tensor=True)
        sim = util.cos_sim(auto_emb, human_emb)  # auto x human

        # Precision: fraction of auto with any human match >= threshold
        auto_max = sim.max(dim=1).values
        prec_matches = (auto_max >= threshold).sum().item()
        precision = prec_matches / len(auto_names) if auto_names else None

        # Recall: fraction of human with any auto match >= threshold
        human_max = sim.max(dim=0).values
        rec_matches = (human_max >= threshold).sum().item()
        recall = rec_matches / len(human_names) if human_names else None

        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0

        return {
            "semantic_precision": round(precision, 4) if precision is not None else None,
            "semantic_recall": round(recall, 4) if recall is not None else None,
            "semantic_f1": round(f1, 4) if f1 is not None else None,
        }
    except Exception:
        return {"semantic_precision": None, "semantic_recall": None, "semantic_f1": None}


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
        print(f"WARN: Coverage evaluator unavailable, continuing without coverage scoring. Details: {e}")
        return None, gt_path


def _extract_run_rows(
    manifest_path: Path,
    payload: Dict[str, Any],
    *,
    coverage_eval: Optional[CoverageEvaluator],
    gt_path: Optional[Path],
    xsd_path: Optional[Path],
    include_failed: bool,
    progress_prefix: str = "",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    manifest_created = payload.get("created_at")
    manifest_provider = payload.get("llm_provider")
    manifest_model = payload.get("llm_model")
    manifest_collection_mode = payload.get("collection_mode")
    runs = payload.get("runs") or []

    total_runs = len(runs)
    for idx, r in enumerate(runs, start=1):
        k = _as_int(r.get("k_value"))
        if k is None:
            continue

        run_id = r.get("run_id")
        repeat = _as_int(r.get("repeat"))
        attempt = _as_int(r.get("attempt"))
        run_model = r.get("llm_model") or r.get("model") or manifest_model or "unknown"
        run_provider = manifest_provider or "unknown"
        run_error = (r.get("error") or "").strip()

        fm_xml_raw = r.get("fm_xml")
        fm_xml_path: Optional[Path] = None
        if fm_xml_raw:
            p = Path(str(fm_xml_raw)).expanduser()
            fm_xml_path = p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()
            if not fm_xml_path.exists():
                fm_xml_path = None

        base_row: Dict[str, Any] = {
            "manifest": str(manifest_path),
            "manifest_created_at": manifest_created,
            "llm_provider": run_provider,
            "model": str(run_model),
            "collection_mode": r.get("collection_mode") or manifest_collection_mode or "unknown",
            "k_value": k,
            "repeat": repeat,
            "attempt": attempt,
            "run_id": run_id,
            "run_error": run_error or None,
            "xml_ok_manifest": r.get("xml_ok"),
            "xml_error_manifest": r.get("xml_error"),
            "fm_xml": str(fm_xml_path) if fm_xml_path else None,
            "meta": r.get("meta"),
            "prompt": r.get("prompt"),
            "evidence": r.get("evidence"),
            "num_evidence_chunks": _as_int(r.get("num_evidence_chunks")),
            "n_results_per_collection_effective": _as_int(r.get("n_results_per_collection_effective")),
            "llm_duration_seconds": _as_float(r.get("llm_duration_seconds")),
            "coverage_score": None,
            "coverage_error": None,
            "semantic_precision": None,
            "semantic_recall": None,
            "semantic_f1": None,
            "feature_count": None,
            "abstract_feature_count": None,
            "concrete_feature_count": None,
            "feature_extract_error": None,
            "wellformed_ok": None,
            "wellformed_error_count": None,
            "wellformed_errors": None,
            "num_duplications": None,
            "num_duplication_groups": None,
            "num_near_duplications": None,
            "num_near_duplication_groups": None,
        }

        if fm_xml_path is None:
            if include_failed:
                rows.append(base_row)
            continue

        wf = validate_feature_model(fm_xml_path, xsd_path)
        base_row["wellformed_ok"] = wf.ok
        base_row["wellformed_error_count"] = len(wf.errors)
        base_row["wellformed_errors"] = wf.errors

        try:
            feats = extract_feature_list(fm_xml_path)
            base_row["feature_count"] = len(feats)
            base_row["abstract_feature_count"] = sum(1 for f in feats if f.feature_type == "abstract")
            base_row["concrete_feature_count"] = sum(1 for f in feats if f.feature_type == "concrete")
        except Exception as e:
            base_row["feature_extract_error"] = str(e)

        base_row.update(_dup_stats(fm_xml_path))

        if coverage_eval is not None and gt_path is not None:
            try:
                base_row["coverage_score"] = coverage_eval.score(gt_path, fm_xml_path, verbose=False)
            except Exception as e:
                base_row["coverage_error"] = str(e)
            # semantic precision/recall/f1 (node-level, same threshold)
            prf = _semantic_prf(
                gt_path,
                fm_xml_path,
                model=coverage_eval.model,
                threshold=coverage_eval.cfg.similarity_threshold,
            )
            base_row.update(prf)

        if include_failed or (base_row["run_error"] is None):
            rows.append(base_row)

        if progress_prefix and (idx == 1 or idx == total_runs or idx % 10 == 0):
            print(
                f"{progress_prefix} processed runs: {idx}/{total_runs}",
                flush=True,
            )

    return rows


def _summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        model = str(r.get("model") or "unknown")
        collection_mode = str(r.get("collection_mode") or "unknown")
        k = _as_int(r.get("k_value"))
        if k is None:
            continue
        grouped[(model, collection_mode, k)].append(r)

    out: List[Dict[str, Any]] = []
    for (model, collection_mode, k), grp in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        runs_total = len(grp)
        runs_with_xml = sum(1 for r in grp if r.get("fm_xml"))
        runs_with_errors = sum(1 for r in grp if r.get("run_error"))
        wf_values = [r.get("wellformed_ok") for r in grp if r.get("wellformed_ok") is not None]
        wellformed_rate = (sum(1 for v in wf_values if v) / len(wf_values)) if wf_values else None

        row = {
            "model": model,
            "collection_mode": collection_mode,
            "k_value": k,
            "runs_total": runs_total,
            "runs_with_xml": runs_with_xml,
            "runs_with_errors": runs_with_errors,
            "coverage_score_mean": _mean([_as_float(r.get("coverage_score")) for r in grp]),
            "semantic_precision_mean": _mean([_as_float(r.get("semantic_precision")) for r in grp]),
            "semantic_recall_mean": _mean([_as_float(r.get("semantic_recall")) for r in grp]),
            "semantic_f1_mean": _mean([_as_float(r.get("semantic_f1")) for r in grp]),
            "feature_count_mean": _mean([_as_float(r.get("feature_count")) for r in grp]),
            "abstract_feature_count_mean": _mean([_as_float(r.get("abstract_feature_count")) for r in grp]),
            "concrete_feature_count_mean": _mean([_as_float(r.get("concrete_feature_count")) for r in grp]),
            "wellformed_rate": round(wellformed_rate, 4) if wellformed_rate is not None else None,
            "llm_duration_seconds_mean": _mean([_as_float(r.get("llm_duration_seconds")) for r in grp]),
            "num_evidence_chunks_mean": _mean([_as_float(r.get("num_evidence_chunks")) for r in grp]),
            "num_duplications_mean": _mean([_as_float(r.get("num_duplications")) for r in grp]),
            "num_duplication_groups_mean": _mean([_as_float(r.get("num_duplication_groups")) for r in grp]),
            "num_near_duplications_mean": _mean([_as_float(r.get("num_near_duplications")) for r in grp]),
        }
        out.append(row)

    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build enriched SS-RAG k-ablation datasets")
    ap.add_argument(
        "--manifests",
        nargs="+",
        default=["results/rag/ss-rgfm/ablation/ss_rgfm_k_runs_*.json"],
        help="Manifest files, directories, or glob patterns.",
    )
    ap.add_argument("--gt", default="", help="Ground-truth XML for coverage scoring (optional).")
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
    ap.add_argument(
        "--require-coverage",
        action="store_true",
        help="Fail if coverage evaluator cannot be initialized.",
    )
    ap.add_argument("--include-failed", action="store_true", help="Keep failed/missing XML runs in output.")
    ap.add_argument("--out-dir", default="results/rag/ss-rgfm/analysis", help="Output directory root.")
    ap.add_argument("--label", default="", help="Optional run label for output folder.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_paths()

    manifest_paths = _resolve_input_paths(args.manifests)
    if not manifest_paths:
        raise FileNotFoundError("No manifest files found from --manifests input.")
    print(f"Found manifest files: {len(manifest_paths)}", flush=True)

    gt_path = Path(args.gt).expanduser().resolve() if args.gt.strip() else None
    xsd_path = Path(args.xsd).expanduser().resolve() if args.xsd.strip() else None
    if xsd_path is not None and not xsd_path.exists():
        xsd_path = None

    if gt_path is not None:
        print(
            f"Initializing coverage evaluator (GT enabled): {gt_path}",
            flush=True,
        )
    else:
        print("Coverage evaluator disabled (no --gt supplied).", flush=True)
    coverage_eval, gt_path = _init_coverage(gt_path, args)
    if gt_path is not None:
        if coverage_eval is None:
            print("Coverage evaluator unavailable; continuing without coverage scores.", flush=True)
        else:
            print("Coverage evaluator ready.", flush=True)

    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    label = args.label.strip() or f"ablation_data_{ts}"
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve() / label)
    print(f"Output directory: {out_dir}", flush=True)

    all_rows: List[Dict[str, Any]] = []
    total_manifests = len(manifest_paths)
    for mi, mf in enumerate(manifest_paths, start=1):
        print(f"[{mi}/{total_manifests}] Processing manifest: {mf}", flush=True)
        payload = json.loads(mf.read_text(encoding="utf-8"))
        before = len(all_rows)
        all_rows.extend(
            _extract_run_rows(
                mf,
                payload,
                coverage_eval=coverage_eval,
                gt_path=gt_path,
                xsd_path=xsd_path,
                include_failed=args.include_failed,
                progress_prefix=f"[{mi}/{total_manifests}]",
            )
        )
        print(
            f"[{mi}/{total_manifests}] Added rows: {len(all_rows) - before}; total rows: {len(all_rows)}",
            flush=True,
        )

    summary_rows = _summarize(all_rows)

    runs_json = out_dir / "ablation_runs_enriched.json"
    runs_csv = out_dir / "ablation_runs_enriched.csv"
    summary_json = out_dir / "ablation_summary_per_k.json"
    summary_csv = out_dir / "ablation_summary_per_k.csv"
    summary_wf_json = out_dir / "ablation_summary_per_k_wf_only.json"
    summary_wf_csv = out_dir / "ablation_summary_per_k_wf_only.csv"
    meta_json = out_dir / "dataset_meta.json"

    runs_json.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    wf_rows = [r for r in all_rows if r.get("wellformed_ok") is True]
    summary_wf_rows = _summarize(wf_rows)
    summary_wf_json.write_text(json.dumps(summary_wf_rows, indent=2), encoding="utf-8")

    run_fields = [
        "manifest",
        "manifest_created_at",
        "llm_provider",
        "model",
        "collection_mode",
        "k_value",
        "repeat",
        "attempt",
        "run_id",
        "fm_xml",
        "num_evidence_chunks",
        "n_results_per_collection_effective",
        "llm_duration_seconds",
        "coverage_score",
        "semantic_precision",
        "semantic_recall",
        "semantic_f1",
        "feature_count",
        "abstract_feature_count",
        "concrete_feature_count",
        "num_duplications",
        "num_duplication_groups",
        "num_near_duplications",
        "num_near_duplication_groups",
        "wellformed_ok",
        "wellformed_error_count",
        "run_error",
        "coverage_error",
        "feature_extract_error",
    ]
    _write_csv(runs_csv, all_rows, run_fields)

    summary_fields = [
        "model",
        "collection_mode",
        "k_value",
        "runs_total",
        "runs_with_xml",
        "runs_with_errors",
        "coverage_score_mean",
        "semantic_precision_mean",
        "semantic_recall_mean",
        "semantic_f1_mean",
        "feature_count_mean",
        "abstract_feature_count_mean",
        "concrete_feature_count_mean",
        "wellformed_rate",
        "llm_duration_seconds_mean",
        "num_evidence_chunks_mean",
        "num_duplications_mean",
        "num_duplication_groups_mean",
        "num_near_duplications_mean",
    ]
    _write_csv(summary_csv, summary_rows, summary_fields)
    _write_csv(summary_wf_csv, summary_wf_rows, summary_fields)

    meta = {
        "created_at": ts,
        "label": label,
        "manifests": [str(p) for p in manifest_paths],
        "ground_truth_xml": str(gt_path) if gt_path else None,
        "xsd": str(xsd_path) if xsd_path else None,
        "runs_count": len(all_rows),
        "summary_rows": len(summary_rows),
        "summary_rows_wellformed_only": len(summary_wf_rows),
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

    print("SUCCESS: Built ablation dataset")
    print(f"Output dir    : {out_dir}")
    print(f"Runs rows     : {len(all_rows)}")
    print(f"Summary rows  : {len(summary_rows)}")
    print(f"Runs CSV      : {runs_csv}")
    print(f"Summary CSV   : {summary_csv}")


if __name__ == "__main__":
    main()
