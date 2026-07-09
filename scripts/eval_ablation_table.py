from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="ablation_summary.json")
    parser.add_argument("--output_csv", type=str, default="ablation_summary.csv")
    args = parser.parse_args()
    rows = []
    for path in sorted(Path(args.runs_dir).rglob("*.json")):
        if "metrics" not in path.name and "eval" not in path.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        data["run_path"] = str(path.parent)
        data.setdefault("ablation", data.get("ablation_mode", path.parent.name))
        rows.append(data)
    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    keys = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    print(json.dumps({"rows": len(rows), "json": str(out_json), "csv": str(out_csv)}, indent=2))


if __name__ == "__main__":
    main()
