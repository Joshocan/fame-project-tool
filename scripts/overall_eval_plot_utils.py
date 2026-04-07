#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except Exception:
    sns = None


METRIC_COLUMNS = [
    "semantic_f1_mean",
    "coverage_score_mean",
    "wellformed_rate",
    "satisfiable_rate",
    "llm_duration_seconds_mean",
    "dead_features_count_mean",
    "feature_count_mean",
]

PIPELINE_ORDER = ["ss_rag", "is_rag", "ss_nonrag", "is_nonrag"]
PIPELINE_LABELS = {
    "ss_rag": "SS-RAG",
    "is_rag": "IS-RAG",
    "ss_nonrag": "SS-NonRAG",
    "is_nonrag": "IS-NonRAG",
}

PIPELINE_COLORS = {
    "ss_rag": "#1f77b4",
    "is_rag": "#4c9ed9",
    "ss_nonrag": "#ffbf00",
    "is_nonrag": "#d95f02",
}

GROUP_COLORS = {
    "RAG": "#1f77b4",
    "Non-RAG": "#ffbf00",
    "Single-stage": "#6baed6",
    "Iterative": "#fd8d3c",
    "Open": "#2ca02c",
    "Proprietary": "#9467bd",
}


def load_summary_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def mean(values: Iterable[Any]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def aggregate_rows(rows: Sequence[dict], group_keys: Sequence[str], metrics: Sequence[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.get(k) for k in group_keys)].append(row)

    out: list[dict] = []
    for key, bucket in buckets.items():
        base = {k: v for k, v in zip(group_keys, key)}
        for metric in metrics:
            base[metric] = mean(as_float(row.get(metric)) for row in bucket)
        out.append(base)
    return out


def aggregate_with_count(
    rows: Sequence[dict],
    group_keys: Sequence[str],
    metrics: Sequence[str],
    *,
    count_label: str = "n",
) -> list[dict]:
    out = aggregate_rows(rows, group_keys, metrics)
    counts: dict[tuple, int] = defaultdict(int)
    for row in rows:
        counts[tuple(row.get(k) for k in group_keys)] += 1
    for row in out:
        row[count_label] = counts[tuple(row.get(k) for k in group_keys)]
    return out


def pipeline_sort_key(row: dict) -> tuple:
    pipeline = str(row.get("pipeline") or "")
    try:
        idx = PIPELINE_ORDER.index(pipeline)
    except ValueError:
        idx = len(PIPELINE_ORDER)
    return (idx, pipeline, str(row.get("model") or ""))


def add_pipeline_label(rows: Sequence[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        clone = dict(row)
        clone["pipeline_label"] = PIPELINE_LABELS.get(str(row.get("pipeline")), str(row.get("pipeline")))
        out.append(clone)
    return out


def rag_group_for_pipeline(pipeline: str) -> str:
    if pipeline in {"ss_rag", "is_rag"}:
        return "RAG"
    if pipeline in {"ss_nonrag", "is_nonrag"}:
        return "Non-RAG"
    return "Other"


def stage_group_for_pipeline(pipeline: str) -> str:
    if pipeline in {"ss_rag", "ss_nonrag"}:
        return "Single-stage"
    if pipeline in {"is_rag", "is_nonrag"}:
        return "Iterative"
    return "Other"


def family_group_for_pipeline(pipeline: str) -> str:
    if pipeline in {"ss_rag", "is_rag"}:
        return "RAG family"
    if pipeline in {"ss_nonrag", "is_nonrag"}:
        return "Non-RAG family"
    return "Other"


def model_type_for_name(model: str) -> str:
    m = (model or "").strip().lower()
    open_prefixes = ("gpt-oss", "deepseek", "glm", "llama", "mistral", "qwen")
    if m.startswith(open_prefixes):
        return "Open"
    return "Proprietary"


def add_model_type(rows: Sequence[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        clone = dict(row)
        clone["model_type"] = model_type_for_name(str(row.get("model") or ""))
        out.append(clone)
    return out


def metric_label(metric: str) -> str:
    mapping = {
        "semantic_f1_mean": "Semantic F1",
        "coverage_score_mean": "Coverage",
        "wellformed_rate": "Well-formed rate",
        "satisfiable_rate": "Satisfiable rate",
        "llm_duration_seconds_mean": "Runtime (s)",
        "dead_features_count_mean": "Dead features",
        "feature_count_mean": "Feature count",
    }
    return mapping.get(metric, metric)


def bar_plot(
    ax,
    rows: Sequence[dict],
    *,
    category_key: str,
    metric_key: str,
    title: str,
    color_fn: Optional[Callable[[dict], str]] = None,
    annotate: bool = True,
) -> None:
    if not rows:
        ax.set_title(f"{title} (no data)")
        return
    xs = list(range(len(rows)))
    labels = [str(row.get(category_key, "")) for row in rows]
    ys = [as_float(row.get(metric_key)) or 0.0 for row in rows]
    colors = [color_fn(row) if color_fn else "#4c78a8" for row in rows]
    bars = ax.bar(xs, ys, color=colors)
    ax.set_xticks(xs, labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric_label(metric_key))
    if annotate:
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2.0, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8)


def grouped_bar_plot(
    ax,
    rows: Sequence[dict],
    *,
    category_key: str,
    series_key: str,
    metric_key: str,
    category_order: Sequence[str],
    series_order: Sequence[str],
    title: str,
    palette: Optional[dict[str, str]] = None,
) -> None:
    if not rows:
        ax.set_title(f"{title} (no data)")
        return
    width = 0.35
    x_positions = list(range(len(category_order)))
    offsets = [(-width / 2.0), (width / 2.0)]
    for series_idx, series_name in enumerate(series_order):
        vals = []
        for cat in category_order:
            match = next((row for row in rows if row.get(category_key) == cat and row.get(series_key) == series_name), None)
            vals.append(as_float(match.get(metric_key)) if match else None)
        yvals = [v if v is not None else 0.0 for v in vals]
        shift = offsets[series_idx] if series_idx < len(offsets) else (series_idx - 0.5) * width
        bars = ax.bar(
            [x + shift for x in x_positions],
            yvals,
            width=width,
            label=series_name,
            color=(palette or GROUP_COLORS).get(series_name, None),
        )
        for bar, y, raw in zip(bars, yvals, vals):
            if raw is not None:
                ax.text(bar.get_x() + bar.get_width() / 2.0, y, f"{raw:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x_positions, category_order)
    ax.set_title(title)
    ax.set_ylabel(metric_label(metric_key))
    ax.legend(fontsize="small")


def scatter_runtime_vs_f1(ax, rows: Sequence[dict], *, title: str, group_key: str) -> None:
    if not rows:
        ax.set_title(f"{title} (no data)")
        return
    seen: set[str] = set()
    for row in rows:
        x = as_float(row.get("llm_duration_seconds_mean"))
        y = as_float(row.get("semantic_f1_mean"))
        if x is None or y is None:
            continue
        group = str(row.get(group_key) or "")
        label = None if group in seen else group
        seen.add(group)
        color = GROUP_COLORS.get(group, PIPELINE_COLORS.get(group, "#4c78a8"))
        ax.scatter(x, y, s=80, color=color, label=label, alpha=0.85, edgecolors="black", linewidths=0.5)
        name = row.get("pipeline_label") or row.get("model") or group
        ax.text(x, y, str(name), fontsize=7, ha="left", va="bottom")
    ax.set_title(title)
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Semantic F1")
    if seen:
        ax.legend(fontsize="small")


def save_manifest(out_dir: Path, tables: Sequence[Path], figures: Sequence[Path]) -> Path:
    manifest = {
        "tables": [str(p) for p in tables],
        "figures": [str(p) for p in figures],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def resolve_data_dir(data_dir: str) -> Path:
    path = Path(data_dir).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")
    return path
