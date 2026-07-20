from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


OFFICIAL_APPEARANCE = ("PSNR", "SSIM", "LPIPS")
LEGACY_PROXY_KEYS = ("render_DINO_similarity", "render_multi_view_consistency")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate legacy and protocol-v2 MeshFleet evaluation results.")
    parser.add_argument("--input_dir", default="outputs/evaluation_suite_metrics")
    parser.add_argument("--output_dir", default="outputs/evaluation_suite_analysis")
    parser.add_argument("--baseline", default="original_trellis")
    parser.add_argument("--worst_views", type=int, default=25)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_records(input_dir)
    if not records:
        raise FileNotFoundError(f"No geovis_slat_metrics.json files under {input_dir}")
    methods = sorted({record["method"] for record in records})
    all_uids = sorted({record["uid"] for record in records})
    grouped = {method: [record for record in records if record["method"] == method] for method in methods}
    aggregates = {
        method: _method_summary(method_records, all_uids)
        for method, method_records in grouped.items()
    }
    paired = {
        method: _paired_summary(grouped.get(args.baseline, []), grouped[method])
        for method in methods
        if method != args.baseline
    }
    view_rows = _view_rows(records)
    view_index_summary = _view_index_summary(view_rows)
    worst_views = sorted(view_rows, key=lambda row: row.get("PSNR", math.inf))[: max(0, args.worst_views)]
    invalid_reasons = _integrity_findings(records)
    result = {
        "analysis_version": "evaluation_suite_audit_v2",
        "input_dir": str(input_dir),
        "record_count": len(records),
        "object_count": len(all_uids),
        "methods": methods,
        "aggregates": aggregates,
        "paired_vs_baseline": paired,
        "view_index_summary": view_index_summary,
        "worst_views": worst_views,
        "integrity_findings": invalid_reasons,
        "legacy_results_are_official": False,
    }
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_records(output_dir / "per_object.csv", records)
    _write_records(output_dir / "per_view.csv", view_rows)
    (output_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _read_records(root: Path) -> List[Dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("geovis_slat_metrics.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            records.append({"uid": path.parts[-4], "method": path.parts[-3], "path": str(path), "parse_error": str(exc)})
            continue
        relative = path.relative_to(root)
        uid = relative.parts[0] if len(relative.parts) >= 1 else "unknown"
        method = relative.parts[1] if len(relative.parts) >= 2 else str(payload.get("ablation", "unknown"))
        record: Dict[str, Any] = {
            "uid": uid,
            "method": method,
            "path": str(path),
            "declared_protocol": payload.get("evaluation_protocol"),
            "declared_official": payload.get("official_metrics"),
            "num_views": payload.get("render_num_views"),
            "has_geometry": "CD" in payload or "Chamfer Distance" in payload,
            "has_legacy_proxies": any(key in payload for key in LEGACY_PROXY_KEYS),
            "view_metrics": payload.get("render_view_metrics", []),
        }
        for metric in (*OFFICIAL_APPEARANCE, "Mask_IoU", "Boundary_F_score", "masked_PSNR", "masked_SSIM", "masked_LPIPS"):
            value = payload.get(f"render_{metric}", payload.get(metric))
            if _is_number(value):
                record[metric] = float(value)
        cd = payload.get("CD", payload.get("Chamfer Distance"))
        fscore = payload.get("F-score")
        if _is_number(cd):
            record["CD"] = float(cd)
        if _is_number(fscore):
            record["F-score"] = float(fscore)
        records.append(record)
    return records


def _method_summary(records: List[Dict[str, Any]], expected_uids: List[str]) -> Dict[str, Any]:
    by_uid = {record["uid"]: record for record in records}
    summary: Dict[str, Any] = {
        "n": len(records),
        "expected_n": len(expected_uids),
        "missing_uids": sorted(set(expected_uids) - set(by_uid)),
        "geometry_n": sum(bool(record.get("has_geometry")) for record in records),
        "declared_official_n": sum(record.get("declared_official") is True for record in records),
    }
    for metric in (*OFFICIAL_APPEARANCE, "CD", "F-score", "Mask_IoU", "Boundary_F_score"):
        values = [record[metric] for record in records if _is_number(record.get(metric))]
        if values:
            summary[metric] = _distribution(values)
            direction = 1 if metric in {"PSNR", "SSIM", "F-score", "Mask_IoU", "Boundary_F_score"} else -1
            worst = min((record for record in records if _is_number(record.get(metric))), key=lambda row: direction * row[metric])
            summary[f"worst_{metric}"] = {"uid": worst["uid"], "value": worst[metric]}
    return summary


def _paired_summary(baseline: List[Dict[str, Any]], candidate: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = {record["uid"]: record for record in baseline}
    cand = {record["uid"]: record for record in candidate}
    common = sorted(set(base) & set(cand))
    result: Dict[str, Any] = {"paired_n": len(common), "uids": common}
    for metric in (*OFFICIAL_APPEARANCE, "CD", "F-score"):
        deltas = [cand[uid][metric] - base[uid][metric] for uid in common if metric in cand[uid] and metric in base[uid]]
        if deltas:
            result[metric] = _distribution(deltas)
    return result


def _view_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        for view in record.get("view_metrics") or []:
            row = {"uid": record["uid"], "method": record["method"], "view": view.get("view")}
            for key, value in view.items():
                if _is_number(value):
                    row[key] = float(value)
            rows.append(row)
    return rows


def _view_index_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['method']}:view_{int(row['view'])}"] .append(row)
    return {
        key: {
            metric: _distribution([row[metric] for row in items if metric in row])
            for metric in OFFICIAL_APPEARANCE
            if any(metric in row for row in items)
        }
        for key, items in sorted(grouped.items())
    }


def _integrity_findings(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings = []
    for record in records:
        reasons = []
        if record.get("has_legacy_proxies"):
            reasons.append("contains nonstandard proxy metrics labeled as DINO/multi-view consistency")
        if not record.get("has_geometry"):
            reasons.append("CD and F-score are absent")
        protocol = str(record.get("declared_protocol") or "")
        if protocol != "meshfleet_heldout_v2":
            reasons.append("not produced by leakage-checked held-out protocol v2")
        if reasons:
            findings.append({"uid": record["uid"], "method": record["method"], "reasons": reasons})
    return findings


def _distribution(values: List[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "n": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "std": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p10": _percentile(ordered, 0.10),
        "max": ordered[-1],
    }


def _percentile(values: List[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def _write_records(path: Path, records: List[Dict[str, Any]]) -> None:
    scalar_rows = [{key: value for key, value in record.items() if isinstance(value, (str, int, float, bool)) or value is None} for record in records]
    fields = sorted({key for row in scalar_rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _markdown_report(result: Dict[str, Any]) -> str:
    lines = [
        "# Evaluation suite audit",
        "",
        "These are legacy diagnostic results, not official target-comparable metrics. The files use conditioning views, "
        "nonstandard SSIM/LPIPS proxies, omit CD/F-score, and predate the TRELLIS export-frame correction.",
        "",
        "| Method | N | PSNR mean | SSIM mean | LPIPS mean | Geometry N |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, summary in result["aggregates"].items():
        lines.append(
            f"| {method} | {summary['n']} | {_mean(summary, 'PSNR')} | {_mean(summary, 'SSIM')} | "
            f"{_mean(summary, 'LPIPS')} | {summary['geometry_n']} |"
        )
    lines += ["", f"Integrity findings: {len(result['integrity_findings'])}/{result['record_count']} records are non-official.", ""]
    return "\n".join(lines)


def _mean(summary: Dict[str, Any], metric: str) -> str:
    value = summary.get(metric, {}).get("mean")
    return f"{value:.6f}" if _is_number(value) else "—"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


if __name__ == "__main__":
    main()
