#!/usr/bin/env python3
"""
Generate ablation summary plots for Centralised (one_collection) vs Federated (per_source) retrieval.

Outputs (saved to --outdir):
  - fig_best_strategy.png : 4-up view vs max_num_chunks
  - fig_topk.png          : 6-up view vs k_value (by strategy)
  - fig_frontier.png      : optional efficiency frontier (quality vs latency)

Data sources:
  - ablation_summary_per_k.json           -> wellformed_rate (all runs)
  - ablation_summary_per_k_wf_only.json   -> quality/latency metrics (well-formed runs only)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import json
import math
from typing import Optional

import matplotlib.pyplot as plt

try:  # use seaborn if available for nicer styling
    import seaborn as sns

    HAS_SNS = True
    sns.set_theme(style="whitegrid")
except Exception:
    HAS_SNS = False


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _as_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def _mean(vals):
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def add_max_num_chunks(df: list[dict], total_sources: int) -> None:
    for row in df:
        mode = row.get("collection_mode", "")
        k = row.get("k_value", 0)
        row["max_num_chunks"] = k if mode == "one_collection" else k * total_sources


def line(ax, df, x, y, title: str, xlabel: Optional[str] = None):
    if not df:
        ax.set_title(f"{title} (no data)")
        return
    # group by collection_mode & model for styling
    for (mode, model), rows in _group(df, ["collection_mode", "model"]).items():
        rows = sorted(rows, key=lambda r: r[x])
        xs = [r[x] for r in rows]
        ys = [r[y] for r in rows]
        label = f"{mode} | {model}"
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(y)
    ax.legend(fontsize="small")


def line_simple(ax, df, x, y, title: str, xlabel: Optional[str] = None):
    if not df:
        ax.set_title(f"{title} (no data)")
        return
    color_map = {"one_collection": "#1f77b4", "per_source": "#ffbf00"}  # blue, amber
    for mode, rows in _group(df, ["collection_mode"]).items():
        rows = sorted(rows, key=lambda r: r[x])
        xs = [r[x] for r in rows]
        ys = [r[y] for r in rows]
        ax.plot(xs, ys, marker="o", label=mode, color=color_map.get(mode))
    ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(y)
    ax.legend(fontsize="small")


def _group(rows: list[dict], keys: list[str]) -> dict[tuple, list[dict]]:
    out: dict[tuple, list[dict]] = {}
    for r in rows:
        k = tuple(r.get(k) for k in keys)
        out.setdefault(k, []).append(r)
    return out


def subplot_grid(nrows, ncols, figsize=(12, 8)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    return fig, axes


def plot_best_strategy(df_all, df_wf, outdir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    line(axes[0], df_wf, "max_num_chunks", "semantic_f1_mean", "Semantic F1 vs chunk budget")
    line(axes[1], df_wf, "max_num_chunks", "coverage_score_mean", "Coverage vs chunk budget")
    line(axes[2], df_wf, "max_num_chunks", "llm_duration_seconds_mean", "Latency vs chunk budget", "max_num_chunks")
    fig.tight_layout()
    outpath = outdir / "fig_best_strategy.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def aggregate(df: list[dict], group_keys: list[str], metrics: list[str]) -> list[dict]:
    from collections import defaultdict

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in df:
        key = tuple(r.get(k) for k in group_keys)
        buckets[key].append(r)

    out: list[dict] = []
    for key, rows in buckets.items():
        base = {k: v for k, v in zip(group_keys, key)}
        for m in metrics:
            vals = [r.get(m) for r in rows if r.get(m) is not None]
            base[m] = sum(vals) / len(vals) if vals else None
        out.append(base)
    return out


def build_topk_all_rows(df_wf: list[dict]) -> list[dict]:
    rows = aggregate(
        df_wf,
        ["collection_mode", "k_value"],
        ["semantic_f1_mean", "coverage_score_mean"],
    )
    return sorted(rows, key=lambda r: (str(r.get("collection_mode")), float(r.get("k_value") or 0)))


def plot_best_strategy_all(df_all, df_wf, outdir: Path):
    # aggregate over models: group by collection_mode + max_num_chunks
    agg_wf = aggregate(df_wf, ["collection_mode", "max_num_chunks"], ["semantic_f1_mean", "coverage_score_mean", "llm_duration_seconds_mean"])

    fig, axes = subplot_grid(2, 2, figsize=(12, 9))
    line_simple(axes[0][0], agg_wf, "max_num_chunks", "semantic_f1_mean", "Semantic F1 vs chunk budget (ALL LLM)")
    line_simple(axes[0][1], agg_wf, "max_num_chunks", "coverage_score_mean", "Coverage vs chunk budget (ALL LLM)")
    line_simple(axes[1][0], agg_wf, "max_num_chunks", "llm_duration_seconds_mean", "Latency vs chunk budget (ALL LLM)", "max_num_chunks")

    # Frontier (ALL LLM) on bottom-right
    if agg_wf:
        ax = axes[1][1]
        modes = sorted({r["collection_mode"] for r in agg_wf})
        colors = {m: c for m, c in zip(modes, plt.cm.tab10.colors)}
        for r in agg_wf:
            ax.scatter(
                r["llm_duration_seconds_mean"],
                r["semantic_f1_mean"],
                s=max(40, min(400, r["max_num_chunks"])),
                color=colors[r["collection_mode"]],
                alpha=0.7,
                edgecolors="k",
                linewidths=0.5,
                label=r["collection_mode"],
            )
        ax.set_xlabel("LLM duration (s)")
        ax.set_ylabel("Semantic F1")
        ax.set_title("Efficiency frontier (ALL LLM, size=chunk budget)")
        handles, labels = ax.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            if l not in uniq:
                uniq[l] = h
        ax.legend(uniq.values(), uniq.keys(), title="Strategy")
    fig.tight_layout()
    outpath = outdir / "fig_best_strategy_all_llm.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_topk(df_all, df_wf, outdir: Path):
    fig, axes = subplot_grid(2, 2, figsize=(12, 8))

    def subset(mode: str, df):
        return [r for r in df if r.get("collection_mode") == mode]

    line(axes[0][0], subset("one_collection", df_wf), "k_value", "semantic_f1_mean", "Semantic F1 vs k (Centralised)", "k_value")
    line(axes[0][1], subset("per_source", df_wf), "k_value", "semantic_f1_mean", "Semantic F1 vs k (Federated)", "k_value")
    line(axes[1][0], subset("one_collection", df_wf), "k_value", "coverage_score_mean", "Coverage vs k (Centralised)", "k_value")
    line(axes[1][1], subset("per_source", df_wf), "k_value", "coverage_score_mean", "Coverage vs k (Federated)", "k_value")

    fig.tight_layout()
    outpath = outdir / "fig_topk.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_topk_all(df_all, df_wf, outdir: Path):
    # aggregate over models: group by collection_mode + k_value
    agg_wf = build_topk_all_rows(df_wf)

    fig, axes = subplot_grid(2, 2, figsize=(12, 8))

    def sub(mode, data):
        return [r for r in data if r.get("collection_mode") == mode]

    line_simple(axes[0][0], sub("one_collection", agg_wf), "k_value", "semantic_f1_mean", "Semantic F1 vs k (Centralised, ALL LLM)", "k_value")
    line_simple(axes[0][1], sub("per_source", agg_wf), "k_value", "semantic_f1_mean", "Semantic F1 vs k (Federated, ALL LLM)", "k_value")
    line_simple(axes[1][0], sub("one_collection", agg_wf), "k_value", "coverage_score_mean", "Coverage vs k (Centralised, ALL LLM)", "k_value")
    line_simple(axes[1][1], sub("per_source", agg_wf), "k_value", "coverage_score_mean", "Coverage vs k (Federated, ALL LLM)", "k_value")

    fig.tight_layout()
    outpath = outdir / "fig_topk_all_llm.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_num_chunks_all(df_wf, outdir: Path, repeat_label: str, repeat_count: int):
    # aggregate over models: group by collection_mode + chunk budget
    grouped: dict[tuple[str, float], list[dict]] = {}
    for row in df_wf:
        mode = row.get("collection_mode")
        num_chunks = row.get("num_evidence_chunks_mean")
        k = row.get("k_value")
        if mode is None or num_chunks is None or k is None:
            continue
        key = (mode, float(num_chunks))
        grouped.setdefault(key, []).append(row)

    points: list[dict] = []
    for (mode, num_chunks), rows in grouped.items():
        points.append(
            {
                "collection_mode": mode,
                "num_chunks": num_chunks,
                "coverage_score_mean": _mean([_as_float(r.get("coverage_score_mean")) for r in rows]),
                "semantic_f1_mean": _mean([_as_float(r.get("semantic_f1_mean")) for r in rows]),
                "k_value_mean": _mean([_as_float(r.get("k_value")) for r in rows]),
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    colors = {"one_collection": "#1f77b4", "per_source": "#ffbf00"}

    ax_main = axes[0][0]
    for mode in sorted({p["collection_mode"] for p in points}):
        mode_pts = sorted([p for p in points if p["collection_mode"] == mode], key=lambda r: r["num_chunks"])
        xs = [p["num_chunks"] for p in mode_pts]
        ys = [p["semantic_f1_mean"] for p in mode_pts]
        ax_main.plot(xs, ys, color=colors.get(mode, "#888888"), marker="o", label=mode, linewidth=1.5)
        for p in mode_pts:
            if p["k_value_mean"] is not None:
                ax_main.text(p["num_chunks"], p["semantic_f1_mean"], f"k={int(p['k_value_mean'])}", fontsize=7)
    ax_main.set_xlabel("Mean retrieved chunks")
    ax_main.set_ylabel("Semantic F1 (mean)")
    ax_main.set_title("All-LLM: budget vs F1 (k labels)")
    handles, labels = ax_main.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax_main.legend(uniq.values(), uniq.keys(), title="Strategy")
    label_text = f"{repeat_label} (runs={repeat_count})"
    ax_main.text(0.95, 0.02, label_text, transform=ax_main.transAxes, ha="right", va="bottom", fontsize=8, color="#444444")

    ax_cov = axes[0][1]
    for mode in sorted({p["collection_mode"] for p in points}):
        mode_pts = sorted([p for p in points if p["collection_mode"] == mode], key=lambda r: r["num_chunks"])
        xs = [p["num_chunks"] for p in mode_pts]
        cov = [p["coverage_score_mean"] for p in mode_pts]
        ax_cov.plot(xs, cov, color=colors.get(mode, "#888888"), marker="s", label=mode, linewidth=1.5)
        for p, y in zip(mode_pts, cov):
            if p["k_value_mean"] is not None:
                ax_cov.text(p["num_chunks"], y, f"k={int(p['k_value_mean'])}", fontsize=7, ha="left")
    ax_cov.set_xlabel("Mean retrieved chunks")
    ax_cov.set_ylabel("Coverage (mean)")
    ax_cov.set_title("All-LLM: budget vs coverage")
    handles, labels = ax_cov.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax_cov.legend(uniq.values(), uniq.keys(), title="Strategy")
    ax_cov.text(0.95, 0.02, label_text, transform=ax_cov.transAxes, ha="right", va="bottom", fontsize=8, color="#444444")

    fig.tight_layout()
    outpath = outdir / "fig_num_chunks_all_llm.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def plot_frontier(df_wf, outdir: Path):
    """Optional efficiency frontier: quality vs latency."""
    if not df_wf:
        return None
    fig, ax = plt.subplots(figsize=(7, 6))
    # color by strategy, size by chunk budget
    modes = sorted({r["collection_mode"] for r in df_wf})
    colors = {m: c for m, c in zip(modes, plt.cm.tab10.colors)}
    for r in df_wf:
        ax.scatter(
            r["llm_duration_seconds_mean"],
            r["semantic_f1_mean"],
            s=max(40, min(400, r["max_num_chunks"])),  # clamp sizes for readability
            color=colors[r["collection_mode"]],
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
            label=r["collection_mode"],
        )
    ax.set_xlabel("LLM duration (s)")
    ax.set_ylabel("Semantic F1")
    ax.set_title("Efficiency frontier (size = chunk budget)")
    # unique legend entries
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), title="Strategy")
    fig.tight_layout()
    outpath = outdir / "fig_frontier.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


def main():
    ap = argparse.ArgumentParser(description="Plot k-ablation summaries.")
    ap.add_argument("--data-dir", default=None, help="Directory containing ablation_summary_per_k*.json files (optional).")
    ap.add_argument("--summary-json", default=None, help="Path to ablation_summary_per_k.json (all runs).")
    ap.add_argument("--summary-wf-json", default=None, help="Path to ablation_summary_per_k_wf_only.json (well-formed only).")
    ap.add_argument(
        "--total-sources",
        type=int,
        default=4,
        help="Number of sources used in per_source retrieval (needed to compute max_num_chunks).",
    )
    ap.add_argument("--outdir", default="ss_k_ablations", help="Directory to save figures.")
    ap.add_argument("--repeat-label", default="Repeats=20", help="Legend text for repeat count on chunk plot.")
    ap.add_argument("--repeat-count", type=int, default=20, help="Run/repeat count to show on the chunk chart.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    summary_json = (
        Path(args.summary_json)
        if args.summary_json
        else (data_dir / "ablation_summary_per_k.json" if data_dir else Path("ss_k_ablations/ablation_summary_per_k.json"))
    )
    summary_wf_json = (
        Path(args.summary_wf_json)
        if args.summary_wf_json
        else (data_dir / "ablation_summary_per_k_wf_only.json" if data_dir else Path("ss_k_ablations/ablation_summary_per_k_wf_only.json"))
    )

    summary_all = load_json(summary_json)
    summary_wf = load_json(summary_wf_json)

    add_max_num_chunks(summary_all, args.total_sources)
    add_max_num_chunks(summary_wf, args.total_sources)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    topk_all_rows = build_topk_all_rows(summary_wf)
    topk_all_table = outdir / "table_topk_all_llm.csv"
    write_csv(
        topk_all_table,
        topk_all_rows,
        ["collection_mode", "k_value", "semantic_f1_mean", "coverage_score_mean"],
    )

    out1 = plot_best_strategy(summary_all, summary_wf, outdir)
    out2 = plot_topk(summary_all, summary_wf, outdir)
    out3 = plot_frontier(summary_wf, outdir)
    out4 = plot_best_strategy_all(summary_all, summary_wf, outdir)
    out5 = plot_topk_all(summary_all, summary_wf, outdir)
    out6 = plot_num_chunks_all(summary_wf, outdir, args.repeat_label, args.repeat_count)

    print("Saved:")
    print("  ", out1)
    print("  ", out2)
    if out3:
        print("  ", out3)
    print("  ", out4)
    print("  ", out5)
    print("  ", out6)
    print("  ", topk_all_table)


if __name__ == "__main__":
    main()
