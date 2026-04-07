#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


PIPELINE_SPECS = {
    "ss_rag": {
        "label": "SS-RAG",
        "top_root": Path("results/rag/ss-rgfm/top_fm"),
    },
    "is_rag": {
        "label": "IS-RAG",
        "top_root": Path("results/rag/is-rgfm/top_fm"),
    },
    "ss_nonrag": {
        "label": "SS-NonRAG",
        "top_root": Path("results/non_rag/ss-nonrag/top_fm"),
    },
    "is_nonrag": {
        "label": "IS-NonRAG",
        "top_root": Path("results/non_rag/is-nonrag/top_fm"),
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compare full pipeline results against top-k selected FMs.")
    ap.add_argument(
        "--overall-csv",
        default="results/analysis/overall_four_pipelines/overall_pipeline_runs_enriched.csv",
        help="Path to overall_pipeline_runs_enriched.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="results/analysis/overall_four_pipelines",
        help="Directory where comparison CSV/Markdown will be written",
    )
    ap.add_argument(
        "--topk",
        nargs="*",
        type=int,
        default=[1, 3, 5],
        help="Top-k sets to compare (default: 1 3 5)",
    )
    ap.add_argument(
        "--include-pooled",
        action="store_true",
        help="Also add pooled ALL-pipelines rows for all/top-k selections",
    )
    ap.add_argument("--verbose", action="store_true", help="Print progress checkpoints")
    return ap.parse_args()


def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print("[compare_overall_vs_topk] " + msg)


def _load_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _summarize_rows(rows: list[dict], *, pipeline: str, selection: str, n_expected: int | None = None) -> dict:
    f1s = [_to_float(r.get("semantic_f1")) for r in rows]
    covs = [_to_float(r.get("coverage_score")) for r in rows]
    f1s = [x for x in f1s if x is not None]
    covs = [x for x in covs if x is not None]
    return {
        "pipeline": pipeline,
        "pipeline_label": PIPELINE_SPECS.get(pipeline, {}).get("label", pipeline),
        "selection": selection,
        "n": len(rows),
        "n_expected": "" if n_expected is None else n_expected,
        "semantic_f1_mean": "" if not f1s else round(mean(f1s), 4),
        "coverage_score_mean": "" if not covs else round(mean(covs), 4),
    }


def _load_topk_rows(top_root: Path, k: int) -> list[dict]:
    path = top_root / f"top_{k}" / "overall" / "top_overall.csv"
    if not path.exists():
        return []
    return _load_csv_rows(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "pipeline",
        "pipeline_label",
        "selection",
        "n",
        "n_expected",
        "semantic_f1_mean",
        "coverage_score_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Overall vs Top-k Comparison\n\n")
        f.write("| Pipeline | Selection | N | Expected N | Semantic F1 | Coverage |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['pipeline_label']} | {row['selection']} | {row['n']} | {row['n_expected']} | "
                f"{row['semantic_f1_mean']} | {row['coverage_score_mean']} |\n"
            )


def main() -> None:
    args = parse_args()
    overall_csv = Path(args.overall_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    topks = sorted(set(int(k) for k in args.topk))

    _log(args.verbose, f"Loading overall runs from {overall_csv}")
    overall_rows = _load_csv_rows(overall_csv)
    _log(args.verbose, f"Loaded {len(overall_rows)} overall rows")

    comparison_rows: list[dict] = []
    pooled_all_rows: list[dict] = []
    pooled_topk_rows: dict[int, list[dict]] = {k: [] for k in topks}

    for pipeline, spec in PIPELINE_SPECS.items():
        pipeline_rows = [
            row for row in overall_rows
            if row.get("pipeline") == pipeline
            and _to_float(row.get("semantic_f1")) is not None
            and _to_float(row.get("coverage_score")) is not None
        ]
        _log(args.verbose, f"{pipeline}: {len(pipeline_rows)} full-set rows with F1/Coverage")
        comparison_rows.append(_summarize_rows(pipeline_rows, pipeline=pipeline, selection="All"))
        pooled_all_rows.extend(pipeline_rows)

        for k in topks:
            top_rows = _load_topk_rows(spec["top_root"], k)
            _log(args.verbose, f"{pipeline}: loaded {len(top_rows)} rows from top_{k}")
            comparison_rows.append(_summarize_rows(top_rows, pipeline=pipeline, selection=f"Top-{k}", n_expected=k))
            pooled_topk_rows[k].extend(top_rows)

    if args.include_pooled:
        pooled = _summarize_rows(pooled_all_rows, pipeline="ALL", selection="All")
        pooled["pipeline_label"] = "ALL"
        comparison_rows.append(pooled)
        for k in topks:
            pooled_row = _summarize_rows(pooled_topk_rows[k], pipeline="ALL", selection=f"Top-{k}", n_expected=4 * k)
            pooled_row["pipeline_label"] = "ALL"
            comparison_rows.append(pooled_row)

    order = {name: i for i, name in enumerate(PIPELINE_SPECS.keys())}

    def sort_key(row: dict) -> tuple:
        pipe = row["pipeline"]
        sel = row["selection"]
        sel_order = 0 if sel == "All" else int(sel.split("-")[1])
        pipe_order = 999 if pipe == "ALL" else order.get(pipe, 998)
        return (pipe_order, sel_order)

    comparison_rows = sorted(comparison_rows, key=sort_key)

    csv_path = out_dir / "overall_vs_topk_comparison.csv"
    md_path = out_dir / "overall_vs_topk_comparison.md"
    _write_csv(csv_path, comparison_rows)
    _write_md(md_path, comparison_rows)

    print(f"Saved table   : {csv_path}")
    print(f"Saved table   : {md_path}")


if __name__ == "__main__":
    main()
