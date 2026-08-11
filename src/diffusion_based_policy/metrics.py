"""Lightweight metric accumulation for training / evaluation loops."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


class RunningMean:
    """Streaming mean and standard error of a scalar stream."""

    def __init__(self) -> None:
        self.n = 0
        self._sum = 0.0
        self._sumsq = 0.0

    def update(self, value: float, count: int = 1) -> None:
        self.n += count
        self._sum += float(value) * count
        self._sumsq += float(value) ** 2 * count

    def update_batch(self, values: Iterable[float]) -> None:
        arr = np.asarray(list(values), dtype=np.float64)
        self.n += arr.size
        self._sum += float(arr.sum())
        self._sumsq += float((arr ** 2).sum())

    @property
    def mean(self) -> float:
        return self._sum / self.n if self.n else float("nan")

    @property
    def var(self) -> float:
        if self.n < 2:
            return float("nan")
        return max(self._sumsq / self.n - self.mean ** 2, 0.0)

    @property
    def stderr(self) -> float:
        if self.n < 2:
            return float("nan")
        return float(np.sqrt(self.var / self.n))

    def as_dict(self) -> dict:
        return {"mean": self.mean, "stderr": self.stderr, "n": self.n}


class MetricLog:
    """Collects named scalar histories for CSV/JSON export."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = defaultdict(list)
        self.steps: list[int] = []

    def log(self, step: int, **values: float) -> None:
        self.steps.append(int(step))
        for k, v in values.items():
            self.history[k].append(float(v))

    def latest(self) -> dict:
        out = {"step": self.steps[-1] if self.steps else None}
        for k, vs in self.history.items():
            out[k] = vs[-1] if vs else None
        return out

    def rows(self) -> list[dict]:
        rows = []
        for i, step in enumerate(self.steps):
            row = {"step": step}
            for k, vs in self.history.items():
                row[k] = vs[i] if i < len(vs) else ""
            rows.append(row)
        return rows
