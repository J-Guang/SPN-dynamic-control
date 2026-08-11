"""Learning-rate schedules for the custom BSDE training loop."""
from __future__ import annotations

import math


def lr_schedule_kwargs(bsde: dict) -> dict:
    """Parse the experiment-level learning-rate schedule block.

    ``cosine`` is the default publication schedule. A plain string is accepted
    for compact configs; a dict can supply ``alpha``/``min_ratio``, warmup,
    decay length, or piecewise values.
    """
    raw = bsde.get("lr_schedule", "cosine")
    if raw is None:
        raw = "cosine"
    if isinstance(raw, str):
        return {"lr_schedule": raw}
    if not isinstance(raw, dict):
        raise TypeError("bsde.lr_schedule must be a string or mapping")

    kind = str(raw.get("type", "cosine"))
    out = {"lr_schedule": kind}
    if "alpha" in raw or "min_ratio" in raw:
        out["lr_min_ratio"] = float(raw.get("alpha", raw.get("min_ratio")))
    if "warmup_steps" in raw:
        out["lr_warmup_steps"] = int(raw["warmup_steps"])
    if "decay_steps" in raw:
        out["lr_decay_steps"] = int(raw["decay_steps"])
    if "boundaries" in raw:
        out["lr_piecewise_boundaries"] = tuple(int(x) for x in raw["boundaries"])
    if "values" in raw:
        out["lr_piecewise_values"] = tuple(float(x) for x in raw["values"])
    return out


def resolve_training_updates(bsde: dict, override_steps: int | None = None,
                             default: int = 4000) -> int:
    """Return the number of optimizer updates for a full training run.

    Publication configs can express repeated training epochs as
    ``train_steps_per_epoch`` times ``train_epochs``. ``train_steps`` remains
    a simple total-update override for compact experiments.
    """
    if override_steps is not None:
        return int(override_steps)
    if "train_steps_per_epoch" in bsde or "train_epochs" in bsde:
        per_epoch = int(bsde.get("train_steps_per_epoch", bsde.get("train_steps", default)))
        epochs = int(bsde.get("train_epochs", 1))
        return per_epoch * epochs
    return int(bsde.get("train_steps", default))


def bound_curriculum_kwargs(bsde: dict) -> dict:
    """Parse the control-bound curriculum block.

    A linear b-ramp uses

        b_eff = min(b, initial_bound + global_step / (40 * log2(1+dim) * pace)).

    The effective bound starts at ``initial_bound`` and ramps linearly to the
    target b. ``initial_bound`` and ``pace`` are configurable; the lower bound is
    fixed at 0.
    """
    raw = bsde.get("bound_curriculum", "none")
    if raw is None:
        raw = "none"
    if isinstance(raw, str):
        return {"bound_curriculum": raw}
    if not isinstance(raw, dict):
        raise TypeError("bsde.bound_curriculum must be a string or mapping")

    out = {"bound_curriculum": str(raw.get("type", "none"))}
    if "pace" in raw:
        out["curriculum_pace"] = float(raw["pace"])
    if "initial_bound" in raw:
        out["curriculum_initial_bound"] = float(raw["initial_bound"])
    return out


def effective_control_bound(
    step: int,
    *,
    target_bound: float,
    dim: int,
    curriculum: str = "none",
    pace: float = 0.0,
    initial_bound: float = 1.0,
) -> float:
    """Return the effective control bound b_eff for the current optimizer update.

    Linear b-ramp (lower bound fixed at 0). The dimension enters as
    ``log2(1 + dim)`` so the ramp length grows mildly with the problem size
    instead of linearly in ``dim``:

        b_eff = min(b, initial_bound + step / (40 * log2(1 + dim) * pace)).

    The effective bound starts at ``initial_bound`` and rises linearly (slope
    1 / (40 * log2(1 + dim) * pace) per step) until it reaches the target b.
    """
    kind = (curriculum or "none").lower()
    b = float(target_bound)
    if kind in {"none", "constant", "off"} or pace <= 0.0:
        return b
    if kind in {"linear", "b_anneal", "anneal", "ramp"}:
        denom = 40.0 * math.log2(1.0 + float(dim)) * float(pace)
        return min(b, float(initial_bound) + float(step) / denom)
    raise ValueError(f"unknown bound_curriculum: {curriculum!r}")


def learning_rate_at(
    step: int,
    *,
    base_lr: float,
    steps: int,
    schedule: str = "cosine",
    min_ratio: float = 1e-2,
    warmup_steps: int = 0,
    decay_steps: int | None = None,
    piecewise_boundaries: tuple[int, ...] = (),
    piecewise_values: tuple[float, ...] = (),
) -> float:
    kind = (schedule or "cosine").lower()
    if kind in {"constant", "none", "fixed"}:
        return float(base_lr)

    if kind == "cosine":
        warmup = max(0, int(warmup_steps))
        if warmup > 0 and step < warmup:
            return float(base_lr) * float(step + 1) / float(warmup)
        if decay_steps is None:
            decay_steps = max(1, int(steps) - warmup)
        progress = min(max((step - warmup) / float(max(1, decay_steps)), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(base_lr) * (float(min_ratio) + (1.0 - float(min_ratio)) * cosine)

    if kind == "piecewise":
        values = tuple(piecewise_values) or (float(base_lr),)
        boundaries = tuple(piecewise_boundaries)
        idx = sum(step >= b for b in boundaries)
        idx = min(idx, len(values) - 1)
        return float(values[idx])

    raise ValueError(f"unknown lr_schedule: {schedule!r}")
