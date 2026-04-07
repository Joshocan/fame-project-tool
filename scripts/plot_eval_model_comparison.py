#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot evaluation 5: model comparison.")
    ap.add_argument("--data-dir", required=True, help="Directory produced by build_overall_pipeline_data.py")
    ap.add_argument("--summary-json", default="", help="Optional explicit overall_pipeline_summary.json path")
    ap.add_argument("--out-dir", default="", help="Optional explicit output directory")
    ap.add_argument("--verbose", action="store_true", help="Print progress checkpoints while generating plots")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        print("[plot_eval_model_comparison] Initializing plotting libraries and utilities...")

    import matplotlib.pyplot as plt

    from overall_eval_plot_utils import (
        GROUP_COLORS,
        add_model_type,
        aggregate_with_count,
        bar_plot,
        ensure_dir,
        load_summary_rows,
        resolve_data_dir,
        save_manifest,
        write_csv,
    )

    def log(msg: str) -> None:
        if args.verbose:
            print("[plot_eval_model_comparison] " + msg)

    data_dir = resolve_data_dir(args.data_dir)
    summary_path = Path(args.summary_json).expanduser().resolve() if args.summary_json.strip() else (data_dir / "overall_pipeline_summary.json")
    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve() if args.out_dir.strip() else (data_dir / "eval_model_comparison"))
    log(f"Resolved data directory: {data_dir}")
    log(f"Output directory: {out_dir}")

    rows = add_model_type(load_summary_rows(summary_path))
    log(f"Loaded {len(rows)} summary rows from {summary_path}")

    model_rows = aggregate_with_count(
        rows,
        ["model", "pipeline", "model_type"],
        [
            "semantic_f1_mean",
            "coverage_score_mean",
            "wellformed_rate",
            "satisfiable_rate",
            "llm_duration_seconds_mean",
        ],
        count_label="summary_rows",
    )
    model_rows = sorted(model_rows, key=lambda row: (str(row.get("model_type")), str(row.get("model")), str(row.get("pipeline"))))

    type_rows = aggregate_with_count(
        rows,
        ["model_type"],
        [
            "semantic_f1_mean",
            "coverage_score_mean",
            "wellformed_rate",
            "satisfiable_rate",
            "llm_duration_seconds_mean",
        ],
        count_label="summary_rows",
    )
    type_rows = sorted(type_rows, key=lambda row: 0 if row.get("model_type") == "Open" else 1)
    log(f"Aggregated into {len(model_rows)} model rows and {len(type_rows)} model-type rows")

    model_table_path = out_dir / "table_model_comparison_by_model.csv"
    grouped_table_path = out_dir / "table_model_comparison_by_type.csv"
    log(f"Writing CSV outputs to {model_table_path} and {grouped_table_path}")
    write_csv(
        model_table_path,
        model_rows,
        [
            "model",
            "pipeline",
            "model_type",
            "semantic_f1_mean",
            "coverage_score_mean",
            "wellformed_rate",
            "satisfiable_rate",
            "llm_duration_seconds_mean",
            "summary_rows",
        ],
    )
    write_csv(
        grouped_table_path,
        type_rows,
        [
            "model_type",
            "semantic_f1_mean",
            "coverage_score_mean",
            "wellformed_rate",
            "satisfiable_rate",
            "llm_duration_seconds_mean",
            "summary_rows",
        ],
    )

    log("Rendering figure...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric, title in [
        (axes[0], "semantic_f1_mean", "Model type: Semantic F1"),
        (axes[1], "coverage_score_mean", "Model type: Coverage"),
        (axes[2], "wellformed_rate", "Model type: Well-formed rate"),
    ]:
        bar_plot(
            ax,
            type_rows,
            category_key="model_type",
            metric_key=metric,
            title=title,
            color_fn=lambda row: GROUP_COLORS.get(str(row.get("model_type")), "#4c78a8"),
        )
    fig.tight_layout()
    fig_path = out_dir / "fig_model_comparison.png"
    log(f"Saving figure to {fig_path}")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    log("Writing manifest...")
    manifest = save_manifest(out_dir, [model_table_path, grouped_table_path], [fig_path])
    print(f"Saved table   : {model_table_path}")
    print(f"Saved table   : {grouped_table_path}")
    print(f"Saved figure  : {fig_path}")
    print(f"Saved manifest: {manifest}")


if __name__ == "__main__":
    main()
