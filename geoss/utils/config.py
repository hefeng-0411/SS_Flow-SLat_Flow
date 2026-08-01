from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
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
    parser.add_argument(
        "--minimum_dataset_passes",
        type=float,
        default=1.0,
        help=(
            "Fail real training when the initial update/batch topology cannot expose this many "
            "nominal dataset passes. Set explicitly to 0 only for smoke tests."
        ),
    )
    parser.add_argument("--fault_tolerant_save_every", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--dist_backend", type=str, default=None)
    parser.add_argument("--dist_url", type=str, default="env://")
    parser.add_argument("--ddp_find_unused_parameters", type=str2bool, default=False)
    add_early_stopping_args(parser)
    return parser


def apply_config_mappings(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    mappings: Dict[str, Any],
    *,
    argv: list[str] | None = None,
) -> None:
    """Apply config values only when the corresponding CLI option was absent.

    Comparing a parsed value with its parser default is insufficient for booleans:
    an explicit ``--early_stop false`` equals the default and used to be silently
    overwritten by ``early_stop: true`` from YAML.  Presence in argv is the actual
    precedence contract.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    supplied = {
        token.split("=", 1)[0]
        for token in argv
        if isinstance(token, str) and token.startswith("--")
    }
    for name, value in mappings.items():
        if value is None or not hasattr(args, name):
            continue
        if f"--{name}" in supplied:
            continue
        if getattr(args, name) == parser.get_default(name):
            setattr(args, name, value)
