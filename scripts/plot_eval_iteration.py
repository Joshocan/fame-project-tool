#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot evaluation 3: iteration effect.")
    ap.add_argument("--data-dir", required=True, help="Directory produced by build_overall_pipeline_data.py")
    ap.add_argument("--summary-json", default="", help="Optional explicit overall_pipeline_summary.json path")
    ap.add_argument("--out-dir", default="", help="Optional explicit output directory")
    ap.add_argument("--verbose", action="store_true", help="Print progress checkpoints while generating plots")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print("[plot_eval_iteration] Initializing plotting libraries and utilities...")

    import matplotlib.pyplot as plt

    from overall_eval_plot_utils import (
        GROUP_COLORS,
        aggregate_with_count,
        ensure_dir,
        family_group_for_pipeline,
        grouped_bar_plot,
        load_summary_rows,
        resolve_data_dir,
        save_manifest,
        stage_group_for_pipeline,
        write_csv,
    )

    def log(msg: str) -> None:
        if args.verbose:
            print("[plot_eval_iteration] " + msg)

    data_dir = resolve_data_dir(args.data_dir)
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json.strip() else (data_dir / "overall_pipeline_summary.json")
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve() if args.out_dir.strip() else (data_dir / "eval_iteration"))
    log(f"Resolved data directory: {data_dir}")
    log(f"Output directory: {out_dir}")

    rows = load_summary_rows(summary_path)
    log(f"Loaded {len(rows)} summary rows from {summary_path}")
    enriched = []
    for row in rows:
        clone = dict(row)
        clone["family_group"] = family_group_for_pipeline(str(row.get("pipeline")))
        clone["stage_group"] = stage_group_for_pipeline(str(row.get("pipeline")))
        enriched.append(clone)

    table_rows = aggregate_with_count(
        [row for row in enriched if row.get("family_group") in {"RAG family", "Non-RAG family"}],
        ["family_group", "stage_group"],
        [
            "semantic_f1_mean",
            "coverage_score_mean",
            "wellformed_rate",
            "satisfiable_rate",
            "llm_duration_seconds_mean",
        ],
        count_label="summary_rows",
    )
    table_rows = sorted(table_rows, key=lambda row: (str(row.get("family_group")), str(row.get("stage_group"))))
    log(f"Aggregated into {len(table_rows)} iteration rows")

    table_path = out_dir / "table_iteration_effect.csv"
    log(f"Writing CSV output to {table_path}")
    write_csv(
        table_path,
        table_rows,
        [
            "family_group",
            "stage_group",
            "semantic_f1_mean",
            "coverage_score_mean",
            "wellformed_rate",
            "satisfiable_rate",
            "llm_duration_seconds_mean",
            "summary_rows",
        ],
    )

    log("Rendering figure...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    grouped_bar_plot(
        axes[0],
        table_rows,
        category_key="family_group",
        series_key="stage_group",
        metric_key="semantic_f1_mean",
        title="Iteration effect: Semantic F1",
        palette=GROUP_COLORS,
    )
    grouped_bar_plot(
        axes[1],
        table_rows,
        category_key="family_group",
        series_key="stage_group",
        metric_key="coverage_score_mean",
        title="Iteration effect: Coverage",
        palette=GROUP_COLORS,
    )
    grouped_bar_plot(
        axes[2],
        table_rows,
        category_key="family_group",
        series_key="stage_group",
        metric_key="llm_duration_seconds_mean",
        title="Iteration effect: Runtime",
        palette=GROUP_COLORS,
    )
    fig.tight_layout()
    fig_path = out_dir / "fig_iteration_effect.png"
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
