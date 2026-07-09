from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


STAGE_SPECS = {
    "stage1_geoss": {
        "dir": "stage1_geoss",
        "log": "train_sparse_ray_geoss.jsonl",
        "title": "Stage 1: Sparse-Ray GeoSS Evidence",
        "loss_metrics": ["loss", "loss_occ", "loss_dice", "loss_free", "loss_conf", "loss_proj", "anchor_sparsity"],
        "quality_metrics": [
            "confidence_mean",
            "confidence_std",
            "ray_valid_mean",
            "occ_score_mean",
            "free_score_mean",
            "conf_error_corr",
            "occ_prob_min",
            "occ_prob_max",
        ],
    },
    "stage2_ss_velocity": {
        "dir": "stage2_ss_velocity",
        "log": "train_sparse_ray_ss_velocity.jsonl",
        "title": "Stage 2: TRELLIS SS Velocity Adapter",
        "loss_metrics": ["loss", "cfm_mse", "velocity_regularization", "prior_preservation", "identity_error"],
        "quality_metrics": ["velocity_norm", "delta_norm", "clipping_ratio"],
    },
    "stage3_geovis_slat": {
        "dir": "stage3_geovis_slat",
        "log": "train_geovis_slat.jsonl",
        "title": "Stage 3: Geo-Visibility SLAT Adapter",
        "loss_metrics": ["loss", "loss_slat_flow", "loss_prior", "loss_velocity"],
        "quality_metrics": ["slat_confidence_mean", "slat_confidence_std", "visibility_mean", "view_weights_std", "delta_norm", "clipping_ratio"],
    },
    "stage4_geovis_slat_joint": {
        "dir": "stage4_geovis_slat_joint",
        "log": "train_geovis_slat.jsonl",
        "title": "Stage 4: Geo-Visibility SLAT Joint",
        "loss_metrics": ["loss", "loss_slat_flow", "loss_prior", "loss_velocity"],
        "quality_metrics": ["slat_confidence_mean", "slat_confidence_std", "visibility_mean", "view_weights_std", "delta_norm", "clipping_ratio"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-style metric plots from the latest SS_Flow training JSONL logs.")
    parser.add_argument("--output_root", type=str, default="outputs/meshfleet_full_4gpu_sequence")
    parser.add_argument("--report_dir", type=str, default=None)
    parser.add_argument("--smooth", type=float, default=0.9, help="EMA smoothing factor in [0, 1).")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--formats", type=str, default="png,pdf", help="Comma-separated output formats, e.g. png,pdf.")
    parser.add_argument("--max_points", type=int, default=4000, help="Downsample long curves for plotting only; CSV keeps all points.")
    parser.add_argument("--html_refresh_seconds", type=int, default=0, help="Set >0 to auto-refresh the generated HTML dashboard.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir) if args.report_dir else output_root / "metrics_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    _setup_matplotlib()
    import matplotlib.pyplot as plt

    records_by_stage = _load_all_stages(output_root)
    if not any(records_by_stage.values()):
        summary = {"output_root": str(output_root), "stages": {}, "warning": "No JSONL training logs were found yet."}
        _write_json(report_dir / "latest_metrics_summary.json", summary)
        (report_dir / "latest_metrics_summary.md").write_text("# Training Metrics\n\nNo JSONL training logs were found yet.\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    formats = [fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip()]
    latest_summary = {}
    image_paths: list[Path] = []

    for stage_name, records in records_by_stage.items():
        if not records:
            continue
        spec = STAGE_SPECS[stage_name]
        stage_dir = report_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(stage_dir / f"{stage_name}_metrics.csv", records)
        latest_summary[stage_name] = _latest_summary(stage_name, records)

        loss_image = _plot_metric_group(
            plt,
            records,
            spec["loss_metrics"],
            title=f"{spec['title']} - Losses",
            ylabel="loss",
            out_base=stage_dir / f"{stage_name}_losses",
            formats=formats,
            smooth=args.smooth,
            max_points=args.max_points,
            dpi=args.dpi,
        )
        quality_image = _plot_metric_group(
            plt,
            records,
            spec["quality_metrics"],
            title=f"{spec['title']} - Geometry / Control Statistics",
            ylabel="metric value",
            out_base=stage_dir / f"{stage_name}_quality",
            formats=formats,
            smooth=args.smooth,
            max_points=args.max_points,
            dpi=args.dpi,
        )
        image_paths.extend([p for p in [loss_image, quality_image] if p is not None])

    overview_loss = _plot_overview(
        plt,
        records_by_stage,
        metric="loss",
        title="Sequential Training Loss Overview",
        out_base=report_dir / "overview_loss",
        formats=formats,
        smooth=args.smooth,
        max_points=args.max_points,
        dpi=args.dpi,
    )
    overview_conf = _plot_overview(
        plt,
        records_by_stage,
        metric=None,
        title="Confidence / Visibility Overview",
        out_base=report_dir / "overview_confidence_visibility",
        formats=formats,
        smooth=args.smooth,
        max_points=args.max_points,
        dpi=args.dpi,
        per_stage_metric={
            "stage1_geoss": "confidence_mean",
            "stage2_ss_velocity": "clipping_ratio",
            "stage3_geovis_slat": "slat_confidence_mean",
            "stage4_geovis_slat_joint": "slat_confidence_mean",
        },
    )
    image_paths.extend([p for p in [overview_loss, overview_conf] if p is not None])

    summary = {
        "output_root": str(output_root),
        "report_dir": str(report_dir),
        "stages": latest_summary,
        "generated_images": [str(path) for path in image_paths],
    }
    _write_json(report_dir / "latest_metrics_summary.json", summary)
    _write_markdown(report_dir / "latest_metrics_summary.md", summary)
    _write_html(report_dir / "metrics_dashboard.html", summary, report_dir, refresh_seconds=args.html_refresh_seconds)
    print(json.dumps(summary, indent=2))


def _setup_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (9.0, 5.2),
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "lines.linewidth": 1.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _load_all_stages(output_root: Path) -> Dict[str, List[dict]]:
    records_by_stage: Dict[str, List[dict]] = {}
    for stage_name, spec in STAGE_SPECS.items():
        log_path = output_root / spec["dir"] / spec["log"]
        records = _read_jsonl(log_path)
        records_by_stage[stage_name] = _dedupe_and_sort(records)
    return records_by_stage


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(_flatten_dict(item))
    return records


def _flatten_dict(item: Mapping, prefix: str = "") -> dict:
    out = {}
    for key, value in item.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten_dict(value, name))
        else:
            out[name] = value
    return out


def _dedupe_and_sort(records: Sequence[dict]) -> List[dict]:
    if not records:
        return []
    keyed = {}
    no_step = []
    for idx, record in enumerate(records):
        step = record.get("step")
        if _is_number(step):
            keyed[int(step)] = record
        else:
            no_step.append((idx, record))
    ordered = [keyed[step] for step in sorted(keyed)]
    ordered.extend(record for _, record in no_step)
    return ordered


def _write_csv(path: Path, records: Sequence[dict]) -> None:
    if not records:
        return
    keys = sorted({key for record in records for key in record.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: _csv_value(record.get(key)) for key in keys})


def _csv_value(value):
    if isinstance(value, (str, int, float)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def _plot_metric_group(
    plt,
    records: Sequence[dict],
    metrics: Sequence[str],
    *,
    title: str,
    ylabel: str,
    out_base: Path,
    formats: Sequence[str],
    smooth: float,
    max_points: int,
    dpi: int,
) -> Path | None:
    available = [metric for metric in metrics if _has_numeric(records, metric)]
    if not available:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
    steps = _series(records, "step")[0]
    for metric in available:
        xs, ys = _series(records, metric)
        if not xs:
            xs = steps[: len(ys)] if steps else list(range(1, len(ys) + 1))
        xs, ys = _downsample(xs, ys, max_points)
        color_line = ax.plot(xs, ys, alpha=0.25, linewidth=1.0)[0]
        ema = _ema(ys, smooth)
        ax.plot(xs, ema, label=metric, color=color_line.get_color(), linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.28)
    return _save_figure(fig, out_base, formats, dpi)


def _plot_overview(
    plt,
    records_by_stage: Mapping[str, Sequence[dict]],
    *,
    metric: str | None,
    title: str,
    out_base: Path,
    formats: Sequence[str],
    smooth: float,
    max_points: int,
    dpi: int,
    per_stage_metric: Mapping[str, str] | None = None,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
    plotted = False
    for stage_name, records in records_by_stage.items():
        if not records:
            continue
        metric_name = per_stage_metric.get(stage_name) if per_stage_metric else metric
        if not metric_name or not _has_numeric(records, metric_name):
            continue
        xs, ys = _series(records, metric_name)
        if not xs:
            xs = list(range(1, len(ys) + 1))
        xs, ys = _downsample(xs, ys, max_points)
        label = f"{stage_name}:{metric_name}" if per_stage_metric else stage_name
        ax.plot(xs, _ema(ys, smooth), label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title(title)
    ax.set_xlabel("training step")
    ax.set_ylabel("metric value")
    ax.legend(ncol=1)
    ax.grid(True, alpha=0.28)
    return _save_figure(fig, out_base, formats, dpi)


def _save_figure(fig, out_base: Path, formats: Sequence[str], dpi: int) -> Path:
    first_path = None
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        if first_path is None and fmt.lower() in {"png", "jpg", "jpeg", "webp"}:
            first_path = path
    if first_path is None:
        first_path = out_base.with_suffix(f".{formats[0]}")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return first_path


def _series(records: Sequence[dict], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for idx, record in enumerate(records):
        value = record.get(key)
        if not _is_number(value):
            continue
        step = record.get("step", idx + 1)
        xs.append(float(step) if _is_number(step) else float(idx + 1))
        ys.append(float(value))
    return xs, ys


def _has_numeric(records: Sequence[dict], key: str) -> bool:
    return any(_is_number(record.get(key)) for record in records)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _ema(values: Sequence[float], smooth: float) -> list[float]:
    if not values:
        return []
    smooth = min(max(float(smooth), 0.0), 0.999)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(smooth * out[-1] + (1.0 - smooth) * float(value))
    return out


def _downsample(xs: Sequence[float], ys: Sequence[float], max_points: int) -> tuple[list[float], list[float]]:
    xs = list(xs)
    ys = list(ys)
    if max_points <= 0 or len(xs) <= max_points:
        return xs, ys
    stride = max(1, math.ceil(len(xs) / max_points))
    return xs[::stride], ys[::stride]


def _latest_summary(stage_name: str, records: Sequence[dict]) -> dict:
    latest = dict(records[-1])
    keys = [
        "step",
        "loss",
        "mode",
        "world_size",
        "per_gpu_batch_size",
        "global_batch_size",
        "confidence_mean",
        "slat_confidence_mean",
        "visibility_mean",
        "ray_valid_mean",
        "delta_norm",
        "clipping_ratio",
        "identity_error",
    ]
    summary = {"num_records": len(records)}
    for key in keys:
        if key in latest:
            summary[key] = latest[key]
    summary["log_stage"] = stage_name
    return summary


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(path: Path, summary: dict) -> None:
    lines = ["# SS_Flow Training Metrics Snapshot", ""]
    lines.append(f"- output_root: `{summary['output_root']}`")
    lines.append(f"- report_dir: `{summary['report_dir']}`")
    lines.append("")
    lines.append("| Stage | Records | Step | Loss | Global Batch | Key Control Metric |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for stage, item in summary["stages"].items():
        key_metric = _format_key_metric(item)
        lines.append(
            "| {stage} | {records} | {step} | {loss} | {batch} | {metric} |".format(
                stage=stage,
                records=item.get("num_records", ""),
                step=item.get("step", ""),
                loss=_fmt_float(item.get("loss")),
                batch=item.get("global_batch_size", ""),
                metric=key_metric,
            )
        )
    lines.append("")
    lines.append("## Generated Figures")
    for image in summary.get("generated_images", []):
        lines.append(f"- `{image}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(path: Path, summary: dict, report_dir: Path, refresh_seconds: int) -> None:
    refresh = f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">' if refresh_seconds > 0 else ""
    rows = []
    for stage, item in summary["stages"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(stage)}</td>"
            f"<td>{item.get('num_records', '')}</td>"
            f"<td>{item.get('step', '')}</td>"
            f"<td>{html.escape(_fmt_float(item.get('loss')))}</td>"
            f"<td>{item.get('global_batch_size', '')}</td>"
            f"<td>{html.escape(_format_key_metric(item))}</td>"
            "</tr>"
        )
    figures = []
    for image in summary.get("generated_images", []):
        image_path = Path(image)
        try:
            rel = image_path.relative_to(report_dir)
        except ValueError:
            rel = image_path
        figures.append(f'<figure><img src="{html.escape(str(rel).replace(chr(92), "/"))}"><figcaption>{html.escape(image_path.stem)}</figcaption></figure>')
    content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  {refresh}
  <title>SS_Flow Training Metrics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    figure {{ margin: 0 0 28px 0; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
    figcaption {{ color: #555; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>SS_Flow Training Metrics Snapshot</h1>
  <p><b>Output root:</b> {html.escape(summary['output_root'])}</p>
  <table>
    <thead><tr><th>Stage</th><th>Records</th><th>Step</th><th>Loss</th><th>Global Batch</th><th>Key Control Metric</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {''.join(figures)}
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _format_key_metric(item: Mapping) -> str:
    for key in ["confidence_mean", "slat_confidence_mean", "visibility_mean", "ray_valid_mean", "delta_norm", "clipping_ratio", "identity_error"]:
        if key in item:
            return f"{key}={_fmt_float(item[key])}"
    return ""


def _fmt_float(value) -> str:
    if not _is_number(value):
        return "" if value is None else str(value)
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
