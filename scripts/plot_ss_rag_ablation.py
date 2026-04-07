#!/usr/bin/env python3
"""Create SS-RAG ablation tables and charts from enriched datasets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


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


def _latest_data_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    return dirs[-1] if dirs else None


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _build_summary_from_runs(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        model = str(r.get("model") or "unknown")
        k = _as_int(r.get("k_value"))
        if k is None:
            continue
        grouped[(model, k)].append(r)

    out: List[Dict[str, Any]] = []
    for (model, k), grp in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        wf_vals = [r.get("wellformed_ok") for r in grp if r.get("wellformed_ok") is not None]
        wellformed_rate = (sum(1 for x in wf_vals if x) / len(wf_vals)) if wf_vals else None
        out.append(
            {
                "model": model,
                "k_value": k,
                "runs_total": len(grp),
                "coverage_score_mean": _mean([_as_float(r.get("coverage_score")) for r in grp]),
                "feature_count_mean": _mean([_as_float(r.get("feature_count")) for r in grp]),
                "abstract_feature_count_mean": _mean([_as_float(r.get("abstract_feature_count")) for r in grp]),
                "concrete_feature_count_mean": _mean([_as_float(r.get("concrete_feature_count")) for r in grp]),
                "wellformed_rate": round(wellformed_rate, 4) if wellformed_rate is not None else None,
                "llm_duration_seconds_mean": _mean([_as_float(r.get("llm_duration_seconds")) for r in grp]),
                "num_evidence_chunks_mean": _mean([_as_float(r.get("num_evidence_chunks")) for r in grp]),
                "num_duplications_mean": _mean([_as_float(r.get("num_duplications")) for r in grp]),
            }
        )
    return out


def _build_rank_table(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in summary_rows:
        grouped[str(r["model"])].append(r)

    out: List[Dict[str, Any]] = []
    for model, rows in sorted(grouped.items()):
        sortable = sorted(
            rows,
            key=lambda r: (_as_float(r.get("coverage_score_mean")) is None, -(_as_float(r.get("coverage_score_mean")) or -1e9)),
        )
        rank = 0
        for r in sortable:
            score = _as_float(r.get("coverage_score_mean"))
            if score is None:
                continue
            rank += 1
            out.append(
                {
                    "model": model,
                    "rank": rank,
                    "k_value": _as_int(r.get("k_value")),
                    "coverage_score_mean": score,
                    "wellformed_rate": _as_float(r.get("wellformed_rate")),
                    "llm_duration_seconds_mean": _as_float(r.get("llm_duration_seconds_mean")),
                    "feature_count_mean": _as_float(r.get("feature_count_mean")),
                    "num_duplications_mean": _as_float(r.get("num_duplications_mean")),
                    "num_evidence_chunks_mean": _as_float(r.get("num_evidence_chunks_mean")),
                }
            )
    return out


def _build_delta_table(summary_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in summary_rows:
        grouped[str(r["model"])].append(r)

    out: List[Dict[str, Any]] = []
    for model, rows in sorted(grouped.items()):
        rows_sorted = sorted(rows, key=lambda r: _as_int(r.get("k_value")) or 0)
        prev = None
        for r in rows_sorted:
            k = _as_int(r.get("k_value"))
            if k is None:
                continue
            if prev is None:
                out.append(
                    {
                        "model": model,
                        "k_value": k,
                        "prev_k_value": None,
                        "delta_coverage_score_mean": None,
                        "delta_feature_count_mean": None,
                        "delta_llm_duration_seconds_mean": None,
                        "delta_num_duplications_mean": None,
                    }
                )
                prev = r
                continue

            def d(field: str) -> Optional[float]:
                cur = _as_float(r.get(field))
                old = _as_float(prev.get(field))
                if cur is None or old is None:
                    return None
                return round(cur - old, 4)

            out.append(
                {
                    "model": model,
                    "k_value": k,
                    "prev_k_value": _as_int(prev.get("k_value")),
                    "delta_coverage_score_mean": d("coverage_score_mean"),
                    "delta_feature_count_mean": d("feature_count_mean"),
                    "delta_llm_duration_seconds_mean": d("llm_duration_seconds_mean"),
                    "delta_num_duplications_mean": d("num_duplications_mean"),
                }
            )
            prev = r
    return out


def _series_by_model(summary_rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, List[Tuple[int, float]]]:
    return _series_by_model_xy(summary_rows, x_field="k_value", y_field=metric)


def _series_by_model_xy(summary_rows: Sequence[Dict[str, Any]], *, x_field: str, y_field: str) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for r in summary_rows:
        model = str(r["model"])
        x = _as_float(r.get(x_field))
        y = _as_float(r.get(y_field))
        if x is None or y is None:
            continue
        out[model].append((x, y))
    for model in out:
        out[model] = sorted(out[model], key=lambda t: t[0])
    return out


def _draw_line_chart(
    *,
    series: Dict[str, List[Tuple[int, float]]],
    title: str,
    y_label: str,
    out_path: Path,
    x_label: str,
) -> None:
    w, h = 1400, 900
    left, top, right, bottom = 120, 80, 340, 110
    plot_w = w - left - right
    plot_h = h - top - bottom
    font = ImageFont.load_default()
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    all_points = [(x, y) for pts in series.values() for x, y in pts]
    if not all_points:
        d.text((40, 40), f"{title}\nNo data", fill="black", font=font)
        img.save(out_path)
        return

    xs = sorted(set(x for x, _ in all_points))
    ys = [y for _, y in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys)
    if y_max <= 0:
        y_max = 1.0
    y_max *= 1.1

    def x_map(x: float) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (x - x_min) * plot_w / (x_max - x_min)

    def y_map(y: float) -> float:
        if y_max == y_min:
            return top + plot_h / 2
        return top + (y_max - y) * plot_h / (y_max - y_min)

    d.rectangle((left, top, left + plot_w, top + plot_h), outline="black", width=2)

    # grid + ticks
    for i in range(6):
        y_val = y_min + (y_max - y_min) * i / 5
        py = y_map(y_val)
        d.line((left, py, left + plot_w, py), fill=(230, 230, 230), width=1)
        d.text((20, py - 7), f"{y_val:.2f}", fill="black", font=font)

    for x in xs:
        px = x_map(x)
        d.line((px, top + plot_h, px, top + plot_h + 8), fill="black", width=1)
        d.text((px - 10, top + plot_h + 15), str(x), fill="black", font=font)

    for idx, (name, pts) in enumerate(sorted(series.items())):
        color = PALETTE[idx % len(PALETTE)]
        mapped = [(x_map(x), y_map(y)) for x, y in pts]
        if len(mapped) > 1:
            d.line(mapped, fill=color, width=3)
        for px, py in mapped:
            d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=color)

    d.text((left, 25), title, fill="black", font=font)
    d.text((left + plot_w // 2 - 30, h - 35), x_label, fill="black", font=font)
    d.text((20, top - 20), y_label, fill="black", font=font)

    # legend
    lx, ly = left + plot_w + 30, top
    d.text((lx, ly), "Legend", fill="black", font=font)
    ly += 25
    for idx, name in enumerate(sorted(series.keys())):
        color = PALETTE[idx % len(PALETTE)]
        d.line((lx, ly + 6, lx + 24, ly + 6), fill=color, width=3)
        d.text((lx + 32, ly), name, fill="black", font=font)
        ly += 20

    img.save(out_path)


def _series_by_model_mode(summary_rows: Sequence[Dict[str, Any]], *, x_field: str, y_field: str) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """
    Returns mapping: model -> {collection_mode -> [(x, y), ...]}.
    """
    out: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in summary_rows:
        model = str(r.get("model", "unknown"))
        mode = str(r.get("collection_mode", "unknown"))
        x = _as_float(r.get(x_field))
        y = _as_float(r.get(y_field))
        if x is None or y is None:
            continue
        out[model][mode].append((x, y))
    # sort each series
    for model in out:
        for mode in out[model]:
            out[model][mode] = sorted(out[model][mode], key=lambda t: t[0])
    return out


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def _series_by_mode_avg(summary_rows: Sequence[Dict[str, Any]], *, x_field: str, y_field: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Average y over all models, grouped by (collection_mode, x).
    """
    buckets: Dict[Tuple[str, float], List[float]] = defaultdict(list)
    for r in summary_rows:
        mode = str(r.get("collection_mode", "unknown"))
        x = _as_float(r.get(x_field))
        y = _as_float(r.get(y_field))
        if x is None or y is None:
            continue
        buckets[(mode, x)].append(y)
    out: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (mode, x), ys in buckets.items():
        if ys:
            out[mode].append((x, round(sum(ys) / len(ys), 4)))
    for mode in out:
        out[mode] = sorted(out[mode], key=lambda t: t[0])
    return out


def _draw_scatter(
    *,
    summary_rows: Sequence[Dict[str, Any]],
    out_path: Path,
    title: str,
    x_field: str,
    x_label: str,
) -> None:
    points = []
    for r in summary_rows:
        x = _as_float(r.get(x_field))
        y = _as_float(r.get("coverage_score_mean"))
        size = _as_float(r.get("feature_count_mean"))
        k = _as_int(r.get("k_value"))
        model = str(r.get("model"))
        if x is None or y is None or size is None or k is None:
            continue
        points.append((x, y, size, k, model))

    w, h = 1400, 900
    left, top, right, bottom = 120, 80, 380, 110
    plot_w = w - left - right
    plot_h = h - top - bottom
    font = ImageFont.load_default()
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    if not points:
        d.text((40, 40), f"{title}\nNo data", fill="black", font=font)
        img.save(out_path)
        return

    x_vals = [p[0] for p in points]
    y_vals = [p[1] for p in points]
    s_vals = [p[2] for p in points]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    if x_min == x_max:
        x_max += 1
    if y_min == y_max:
        y_max += 1

    def x_map(x: float) -> float:
        return left + (x - x_min) * plot_w / (x_max - x_min)

    def y_map(y: float) -> float:
        return top + (y_max - y) * plot_h / (y_max - y_min)

    s_min, s_max = min(s_vals), max(s_vals)

    def r_map(s: float) -> float:
        if s_min == s_max:
            return 9.0
        return 5 + (s - s_min) * 12 / (s_max - s_min)

    d.rectangle((left, top, left + plot_w, top + plot_h), outline="black", width=2)
    d.text((left, 25), title, fill="black", font=font)
    d.text((left + plot_w // 2 - 80, h - 35), x_label, fill="black", font=font)
    d.text((20, top - 20), "coverage_score_mean", fill="black", font=font)

    ks = sorted(set(p[3] for p in points))
    k_color = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(ks)}

    for x, y, s, k, model in points:
        px, py = x_map(x), y_map(y)
        rr = r_map(s)
        color = k_color[k]
        d.ellipse((px - rr, py - rr, px + rr, py + rr), fill=color, outline=(50, 50, 50))
        d.text((px + rr + 2, py - 7), model, fill="black", font=font)

    # legends
    lx, ly = left + plot_w + 30, top
    d.text((lx, ly), "Color = k", fill="black", font=font)
    ly += 22
    for k in ks:
        c = k_color[k]
        d.rectangle((lx, ly, lx + 14, ly + 14), fill=c, outline="black")
        d.text((lx + 22, ly), f"k={k}", fill="black", font=font)
        ly += 20

    ly += 20
    d.text((lx, ly), "Size = feature_count_mean", fill="black", font=font)

    img.save(out_path)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create SS-RAG ablation tables and charts")
    ap.add_argument("--data-dir", default="", help="Output folder from build_ss_k_rag_ablation_data.py")
    ap.add_argument("--runs-json", default="", help="Optional explicit ablation_runs_enriched.json")
    ap.add_argument("--summary-json", default="", help="Optional explicit ablation_summary_per_k.json")
    ap.add_argument("--out-dir", default="", help="Output report directory (default: <data-dir>/report)")
    ap.add_argument(
        "--x-field",
        default="k_value",
        choices=["k_value", "num_evidence_chunks_mean"],
        help="X-axis field for plots (default: k_value; use num_evidence_chunks_mean to plot by chunk budget).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root_default = Path("results/rag/ss-rgfm/analysis").resolve()

    if args.data_dir.strip():
        data_dir = Path(args.data_dir).expanduser().resolve()
    else:
        latest = _latest_data_dir(root_default)
        if latest is None:
            raise FileNotFoundError("No data directory found. Run build_ss_k_rag_ablation_data.py first.")
        data_dir = latest

    runs_json = Path(args.runs_json).expanduser().resolve() if args.runs_json.strip() else (data_dir / "ablation_runs_enriched.json")
    summary_json = (
        Path(args.summary_json).expanduser().resolve() if args.summary_json.strip() else (data_dir / "ablation_summary_per_k.json")
    )
    if not runs_json.exists():
        raise FileNotFoundError(f"Runs JSON not found: {runs_json}")
    print(f"Loading runs dataset: {runs_json}", flush=True)

    runs_rows = _load_rows(runs_json)
    if summary_json.exists():
        print(f"Loading summary dataset: {summary_json}", flush=True)
        summary_rows = _load_rows(summary_json)
    else:
        print("Summary dataset missing; rebuilding summary from run rows.", flush=True)
        summary_rows = _build_summary_from_runs(runs_rows)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir.strip() else ensure_output_dir(data_dir / "report")
    out_dir = ensure_output_dir(out_dir)
    print(f"Report output directory: {out_dir}", flush=True)

    # tables
    print("Writing report tables...", flush=True)
    per_k_summary = out_dir / "per_k_summary_table.csv"
    per_run_raw = out_dir / "per_run_raw_table.csv"
    rank_csv = out_dir / "rank_table_by_coverage.csv"
    delta_csv = out_dir / "delta_table.csv"

    summary_fields = [
        "model",
        "collection_mode",
        "k_value",
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
        "runs_total",
        "runs_with_xml",
        "runs_with_errors",
    ]
    run_fields = [
        "run_id",
        "model",
        "llm_provider",
        "collection_mode",
        "k_value",
        "repeat",
        "attempt",
        "coverage_score",
        "semantic_precision",
        "semantic_recall",
        "semantic_f1",
        "feature_count",
        "abstract_feature_count",
        "concrete_feature_count",
        "wellformed_ok",
        "wellformed_error_count",
        "num_duplications",
        "num_evidence_chunks",
        "n_results_per_collection_effective",
        "llm_duration_seconds",
        "run_error",
        "coverage_error",
        "feature_extract_error",
        "fm_xml",
    ]
    _write_csv(per_k_summary, summary_rows, summary_fields)
    _write_csv(per_run_raw, runs_rows, run_fields)

    rank_rows = _build_rank_table(summary_rows)
    delta_rows = _build_delta_table(summary_rows)
    _write_csv(
        rank_csv,
        rank_rows,
        [
            "model",
            "rank",
            "k_value",
            "coverage_score_mean",
            "wellformed_rate",
            "llm_duration_seconds_mean",
            "feature_count_mean",
            "num_duplications_mean",
            "num_evidence_chunks_mean",
        ],
    )
    _write_csv(
        delta_csv,
        delta_rows,
        [
            "model",
            "k_value",
            "prev_k_value",
            "delta_coverage_score_mean",
            "delta_feature_count_mean",
            "delta_llm_duration_seconds_mean",
            "delta_num_duplications_mean",
        ],
    )

    # charts
    print("Rendering charts...", flush=True)
    x_field = args.x_field
    x_label = x_field

    _draw_line_chart(
        series=_series_by_model_xy(summary_rows, x_field=x_field, y_field="coverage_score_mean"),
        title=f"{x_label} vs coverage_score_mean",
        y_label="coverage_score_mean",
        x_label=x_label,
        out_path=out_dir / f"line_{x_field}_vs_coverage_score_mean.png",
    )

    feature_series = {}
    for model, pts in _series_by_model_xy(summary_rows, x_field=x_field, y_field="feature_count_mean").items():
        feature_series[f"{model} total"] = pts
    for model, pts in _series_by_model_xy(summary_rows, x_field=x_field, y_field="abstract_feature_count_mean").items():
        feature_series[f"{model} abstract"] = pts
    for model, pts in _series_by_model_xy(summary_rows, x_field=x_field, y_field="concrete_feature_count_mean").items():
        feature_series[f"{model} concrete"] = pts
    _draw_line_chart(
        series=feature_series,
        title=f"{x_label} vs feature_count_mean (total/abstract/concrete)",
        y_label="feature counts mean",
        x_label=x_label,
        out_path=out_dir / f"line_{x_field}_vs_feature_counts_mean.png",
    )

    _draw_line_chart(
        series=_series_by_model_xy(summary_rows, x_field=x_field, y_field="wellformed_rate"),
        title=f"{x_label} vs wellformed_rate",
        y_label="wellformed_rate",
        x_label=x_label,
        out_path=out_dir / f"line_{x_field}_vs_wellformed_rate.png",
    )
    _draw_line_chart(
        series=_series_by_model_xy(summary_rows, x_field=x_field, y_field="llm_duration_seconds_mean"),
        title=f"{x_label} vs llm_duration_seconds_mean",
        y_label="seconds",
        x_label=x_label,
        out_path=out_dir / f"line_{x_field}_vs_llm_duration_seconds_mean.png",
    )
    _draw_line_chart(
        series=_series_by_model_xy(summary_rows, x_field=x_field, y_field="num_evidence_chunks_mean"),
        title=f"{x_label} vs num_evidence_chunks_mean",
        y_label="chunks",
        x_label=x_label,
        out_path=out_dir / f"line_{x_field}_vs_num_evidence_chunks_mean.png",
    )
    _draw_line_chart(
        series=_series_by_model_xy(summary_rows, x_field=x_field, y_field="num_duplications_mean"),
        title=f"{x_label} vs num_duplications_mean",
        y_label="duplications",
        x_label=x_label,
        out_path=out_dir / f"line_{x_field}_vs_num_duplications_mean.png",
    )
    _draw_scatter(
        summary_rows=summary_rows,
        out_path=out_dir / f"scatter_{x_field}_vs_coverage_size_feature_color_k.png",
        title=f"{x_label} vs coverage (size=feature_count_mean, color=k)",
        x_field=x_field if x_field != "k_value" else "llm_duration_seconds_mean",
        x_label=x_label if x_field != "k_value" else "llm_duration_seconds_mean",
    )

    # Tier 1 decision charts (per model, lines = collection_mode, x = evidence chunks)
    tier1_x = "num_evidence_chunks_mean"
    tier1_metrics = [
        "semantic_f1_mean",
        "semantic_precision_mean",
        "semantic_recall_mean",
        "coverage_score_mean",
        "wellformed_rate",
    ]
    tier1_plots: List[str] = []
    for metric in tier1_metrics:
        per_model = _series_by_model_mode(summary_rows, x_field=tier1_x, y_field=metric)
        for model, series in per_model.items():
            safe_model = _slug(model)
            fname = f"line_{safe_model}_{tier1_x}_vs_{metric}.png"
            _draw_line_chart(
                series=series,
                title=f"{model}: {tier1_x} vs {metric}",
                y_label=metric,
                x_label=tier1_x,
                out_path=out_dir / fname,
            )
            tier1_plots.append(fname)

    # Mode-averaged plot (both strategies averaged over all models)
    mode_avg_metric = "semantic_f1_mean"
    mode_avg_series = _series_by_mode_avg(summary_rows, x_field=tier1_x, y_field=mode_avg_metric)
    fname_mode_avg = f"line_mode_avg_{tier1_x}_vs_{mode_avg_metric}.png"
    _draw_line_chart(
        series=mode_avg_series,
        title=f"Avg across models: {tier1_x} vs {mode_avg_metric}",
        y_label=mode_avg_metric,
        x_label=tier1_x,
        out_path=out_dir / fname_mode_avg,
    )
    tier1_plots.append(fname_mode_avg)

    summary = {
        "data_dir": str(data_dir),
        "runs_json": str(runs_json),
        "summary_json": str(summary_json) if summary_json.exists() else None,
        "report_dir": str(out_dir),
        "tables": {
            "per_k_summary": str(per_k_summary),
            "per_run_raw": str(per_run_raw),
            "rank": str(rank_csv),
            "delta": str(delta_csv),
        },
        "plots": [
            f"line_{x_field}_vs_coverage_score_mean.png",
            f"line_{x_field}_vs_feature_counts_mean.png",
            f"line_{x_field}_vs_wellformed_rate.png",
            f"line_{x_field}_vs_llm_duration_seconds_mean.png",
            f"line_{x_field}_vs_num_evidence_chunks_mean.png",
            f"line_{x_field}_vs_num_duplications_mean.png",
            f"scatter_{x_field}_vs_coverage_size_feature_color_k.png",
        ] + tier1_plots,
    }
    (out_dir / "report_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("SUCCESS: Generated ablation report", flush=True)
    print(f"Report dir: {out_dir}", flush=True)
    print(f"Per-k table: {per_k_summary}", flush=True)
    print(f"Per-run table: {per_run_raw}", flush=True)
    print(f"Rank table: {rank_csv}", flush=True)
    print(f"Delta table: {delta_csv}", flush=True)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()
