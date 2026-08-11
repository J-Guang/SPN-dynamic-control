"""Small JSON/CSV helpers with numpy-aware encoding."""
from __future__ import annotations

import csv
import json
import os
from typing import Any

import numpy as np


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any, indent: int = 2) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_default)
    return path


def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_csv(path: str, rows: list[dict], fieldnames: list[str] | None = None) -> str:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    if not rows:
        open(path, "w").close()
        return path
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return path


def append_csv_row(path: str, row: dict, fieldnames: list[str]) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
