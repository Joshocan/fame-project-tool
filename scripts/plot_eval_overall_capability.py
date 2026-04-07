#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot evaluation 1: overall capability.")
    ap.add_argument("--data-dir", required=True, help="Directory produced by build_overall_pipeline_data.py")
    ap.add_argument("--summary-json", default="", help="Optional explicit overall_pipeline_summary.json path")
    ap.add_argument("--out-dir", default="", help="Optional explicit output directory")
    ap.add_argument("--verbose", action="store_true", help="Print progress checkpoints while generating plots")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print("[plot_eval_overall_capability] Initializing plotting libraries and utilities...")

    import matplotlib.pyplot as plt

    from overall_eval_plot_utils import (
        PIPELINE_COLORS,
        add_pipeline_label,
        aggregate_with_count,
        bar_plot,
        ensure_dir,
        load_summary_rows,
        pipeline_sort_key,
        resolve_data_dir,
        save_manifest,
        write_csv,
    )

    def log(msg: str) -> None:
        if args.verbose:
            print("[plot_eval_overall_capability] " + msg)

    data_dir = resolve_data_dir(args.data_dir)
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json.strip() else (data_dir / "overall_pipeline_summary.json")
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve() if args.out_dir.strip() else (data_dir / "eval_overall_capability"))
    log(f"Resolved data directory: {data_dir}")
    log(f"Output directory: {out_dir}")

    rows = load_summary_rows(summary_path)
    log(f"Loaded {len(rows)} summary rows from {summary_path}")
    table_rows = aggregate_with_count(
        rows,
        ["pipeline"],
        [
            "semantic_f1_mean",
            "coverage_score_mean",
            "wellformed_rate",
            "satisfiable_rate",
            "llm_duration_seconds_mean",
        ],
        count_label="summary_rows",
    )
    table_rows = add_pipeline_label(sorted(table_rows, key=pipeline_sort_key))
    log(f"Aggregated into {len(table_rows)} pipeline rows")

    table_path = out_dir / "table_overall_capability.csv"
    log(f"Writing CSV output to {table_path}")
    write_csv(
        table_path,
        table_rows,
        [
            "pipeline",
            "pipeline_label",
            "semantic_f1_mean",
            "coverage_score_mean",
            "wellformed_rate",
            "satisfiable_rate",
            "llm_duration_seconds_mean",
            "summary_rows",
        ],
    )

    log("Rendering figure...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    bar_plot(
        axes[0][0],
        table_rows,
        category_key="pipeline_label",
        metric_key="semantic_f1_mean",
        title="Overall capability: Semantic F1 by pipeline",
        color_fn=lambda row: PIPELINE_COLORS.get(str(row.get("pipeline")), "#4c78a8"),
    )
    bar_plot(
        axes[0][1],
        table_rows,
        category_key="pipeline_label",
        metric_key="coverage_score_mean",
        title="Overall capability: Coverage by pipeline",
        color_fn=lambda row: PIPELINE_COLORS.get(str(row.get("pipeline")), "#4c78a8"),
    )
    bar_plot(
        axes[1][0],
        table_rows,
        category_key="pipeline_label",
        metric_key="wellformed_rate",
        title="Overall capability: Well-formed rate by pipeline",
        color_fn=lambda row: PIPELINE_COLORS.get(str(row.get("pipeline")), "#4c78a8"),
    )
    bar_plot(
        axes[1][1],
        table_rows,
        category_key="pipeline_label",
        metric_key="satisfiable_rate",
        title="Overall capability: Satisfiable rate by pipeline",
        color_fn=lambda row: PIPELINE_COLORS.get(str(row.get("pipeline")), "#4c78a8"),
    )
    fig.tight_layout()
    fig_path = out_dir / "fig_overall_capability.png"
    log(f"Saving figure to {fig_path}")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    log("Writing manifest...")
    manifest = save_manifest(out_dir, [table_path], [fig_path])
    print(f"Saved table   : {table_path}")
    print(f"Saved figure  : {fig_path}")
    print(f"Saved manifest: {manifest}")


if __name__ == "__main__":
    main()
