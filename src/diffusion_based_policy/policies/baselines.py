"""Baseline gradient/policy functions for the prelimit simulator.

The prelimit simulator drives any policy through a gradient function g(z) plugged
into the BCP index pi_j (math_foundation.md Section 5). Baselines express simple
priority rules as effective gradients:

  * ``constant_gradient`` g(z) = htilde  -> static holding-cost priority,
  * ``linear_gradient``   g(z) = htilde o z -> max-weight-like priority,

and ``baseline_gradient`` returns the default smoke baseline. A trained BSDE
model supplies its own G_phi via diffusion_based_policy.bsde.evaluator.BSDEEvaluator.gradient_fn.
"""
from __future__ import annotations

import numpy as np

from ..bcp import BCPModel


def constant_gradient(model: BCPModel):
    h = model.h_tilde.astype(np.float64)

    def fn(z):
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        return np.broadcast_to(h, z.shape).copy()

    return fn


def linear_gradient(model: BCPModel):
    """g(z) = htilde o z (holding-weighted max-pressure / "MP-h" index when fed to pi_j)."""
    h = model.h_tilde.astype(np.float64)

    def fn(z):
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        return h[None, :] * z

    return fn


def identity_gradient(model: BCPModel):
    """g(z) = z (the unweighted max-pressure / "MP" index when fed to pi_j)."""
    def fn(z):
        return np.atleast_2d(np.asarray(z, dtype=np.float64)).copy()

    return fn


def zero_gradient(model: BCPModel):
    def fn(z):
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        return np.zeros_like(z)

    return fn


BASELINES = {
    "identity": identity_gradient,   # MP    : g = z
    "linear": linear_gradient,       # MP-h  : g = h o z
    "constant": constant_gradient,
    "zero": zero_gradient,
}


def baseline_gradient(model: BCPModel, kind: str = "linear"):
    """Return a baseline gradient function g(z) -> g (vectorized over batch)."""
    if kind not in BASELINES:
        raise ValueError(f"unknown baseline '{kind}'; choose from {list(BASELINES)}")
    return BASELINES[kind](model)
