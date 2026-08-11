"""Brownian cost scaling (math_foundation.md Section 2.3).

The Brownian objective uses tilde-costs that are limits of prelimit coefficients:

    h_i^{(n)}      -> tilde h_i = h_i
    c_k^{I,(n)}/n  -> tilde c_k = c_k^I            (processing idleness)
    c_k^{D,(n)}/n  -> c_k^D,  tilde c_k = mu_bar_k c_k^D   (input rejection)

This module isolates the conversion so the scaling rule is testable on its own.
"""
from __future__ import annotations

import numpy as np

from .bcp import BCPModel


def brownian_rejection_costs(model: BCPModel) -> dict:
    """Return the per-input-resource rejection conversion.

    Config holds the prelimit per-job charge c^{D,(n)}; the Brownian limit is
    bar c^D = c^{D,(n)}/n.  For each input resource k:
        c_D_prelimit = model.spec.rejection_cost[k]   (prelimit per-job c^{D,(n)})
        bar c^D_k    = c_D_prelimit / n               (Brownian limit)
        mu_bar_k     = input opportunity rate
        tilde c_k    = mu_bar_k bar c^D_k
    """
    spec = model.spec
    mu_bar = model.input_opportunity_rate
    cD_n = spec.rejection_cost
    n = model.params.n
    out = {}
    for k in spec.input_resources:
        bar_cD = float(cD_n[k] / n)
        out[spec.resource_labels[k]] = {
            "c_D_prelimit": float(cD_n[k]),
            "c_D": bar_cD,
            "mu_bar": float(mu_bar[k]),
            "c_tilde": float(mu_bar[k] * bar_cD),
        }
    return out


def prelimit_rejection_charge(model: BCPModel) -> np.ndarray:
    """Prelimit per-rejected-job charge c_k^{D,(n)} (= config rejection_cost).

    The prelimit simulator charges this per rejected job; dividing simulation
    cost by n^{3/2} then matches the Brownian objective.
    """
    return model.spec.rejection_cost.copy()


def verify_rejection_scaling(model: BCPModel, atol: float = 1e-12) -> dict:
    """Confirm tilde c on input K-rows equals mu_bar o c^D."""
    spec = model.spec
    mu_bar = model.input_opportunity_rate
    # c_tilde is ordered by K rows; map input K-rows back to resources.
    rows = model._K_rows  # noqa: SLF001 - intentional internal access
    c_tilde = model.c_tilde
    checks = []
    ok = True
    for (kind, idx), value in zip(rows, c_tilde):
        if kind == "resource" and spec.is_input[idx]:
            expected = float(mu_bar[idx] * spec.rejection_cost[idx] / model.params.n)
            match = abs(value - expected) <= atol
            ok = ok and match
            checks.append({
                "resource": spec.resource_labels[idx],
                "c_tilde": float(value),
                "expected": expected,
                "ok": bool(match),
            })
    return {"ok": bool(ok), "checks": checks}
