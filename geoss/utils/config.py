from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from geoss.utils.early_stopping import add_early_stopping_args


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def load_config(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml

            data = yaml.safe_load(text)
            return data or {}
        except Exception:
            return {}


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dry_run", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--steps_are_total", type=str2bool, default=False)
    parser.add_argument("--fault_tolerant_save_every", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--dist_backend", type=str, default=None)
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--ddp_find_unused_parameters", type=str2bool, default=False)
    add_early_stopping_args(parser)
    return parser
