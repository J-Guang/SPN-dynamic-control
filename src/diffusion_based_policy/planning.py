"""Static planning problem and critical-point verification.

math_foundation.md Section 1.5 (static planning LP) and Section 2.1 (critical
pair (mu*, beta)).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .bcp import BCPModel
from .network import NetworkSpec


def flow_balance_residual(P: np.ndarray, rates: np.ndarray, x: np.ndarray) -> np.ndarray:
    """P diag(rates) x  (= 0 under flow balance)."""
    return P @ (rates * x)


def static_planning_lp(
    spec: NetworkSpec,
    rates: np.ndarray | None = None,
    pin_inputs: bool = False,
    nominal_allocation: np.ndarray | None = None,
) -> dict:
    """Solve the Section 1.5 static planning LP.

        min rho   s.t.  P diag(mu) x = 0,
                        (A x)_k <= rho      for processing k,
                        sum_{j in B_k} A_kj x_j = 1   for no-rejection input k,
                        sum_{j in B_k} A_kj x_j <= 1  for rejection input k,
                        0 <= x <= 1.

    With ``pin_inputs=True`` the rejection-allowed input rows are also pinned to
    their nominal usage (sum A_kj x_j = sum A_kj beta_j) so the LP reports the
    critical loading rho*=1 instead of the trivial reject-everything rho*=0 that
    a rejection model otherwise admits.

    Returns a dict with rho_star, x, utilization, and the solver status.
    """
    rates = spec.rates if rates is None else np.asarray(rates, dtype=np.float64)
    P, A = spec.buffer_change, spec.resource_use
    I, J = P.shape
    K = A.shape[0]
    nvar = J + 1  # x (J) and rho

    c = np.zeros(nvar)
    c[-1] = 1.0

    # Equality: flow balance P diag(mu) x = 0
    A_eq = [np.concatenate([P[i] * rates, [0.0]]) for i in range(I)]
    b_eq = [0.0] * I
    # Equality: no-rejection input rows fully used
    for k in spec.no_rejection_resources:
        A_eq.append(np.concatenate([A[k], [0.0]]))
        b_eq.append(1.0)
    # Optionally pin rejection-allowed input rows to nominal usage.
    if pin_inputs and nominal_allocation is not None:
        beta = np.asarray(nominal_allocation, dtype=np.float64)
        for k in spec.input_resources:
            if k in spec.no_rejection_resources:
                continue
            A_eq.append(np.concatenate([A[k], [0.0]]))
            b_eq.append(float(A[k] @ beta))

    A_ub, b_ub = [], []
    # processing utilization <= rho
    for k in spec.processing_resources:
        row = np.concatenate([A[k], [-1.0]])
        A_ub.append(row)
        b_ub.append(0.0)
    # rejection-allowed input rows <= 1
    for k in spec.input_resources:
        if k in spec.no_rejection_resources:
            continue
        A_ub.append(np.concatenate([A[k], [0.0]]))
        b_ub.append(1.0)

    bounds = [(0.0, 1.0)] * J + [(0.0, None)]
    res = linprog(
        c,
        A_ub=np.asarray(A_ub) if A_ub else None,
        b_ub=np.asarray(b_ub) if b_ub else None,
        A_eq=np.asarray(A_eq) if A_eq else None,
        b_eq=np.asarray(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    x = res.x[:J] if res.success else np.full(J, np.nan)
    rho = float(res.x[-1]) if res.success else np.nan
    util = A @ x if res.success else np.full(K, np.nan)
    return {
        "rho_star": rho,
        "x": x,
        "utilization": util,
        "success": bool(res.success),
        "status": res.message,
    }


def verify_critical_pair(model: BCPModel, atol: float = 1e-9) -> dict:
    """Check that (mu*, beta) satisfy flow balance and critical loading."""
    spec = model.spec
    P = spec.buffer_change
    mu_star = model.params.critical_rates
    beta = model.params.nominal_allocation

    fb = flow_balance_residual(P, mu_star, beta)
    util = spec.resource_use @ beta

    crit = list(model.params.critical_resources)
    crit_load = {spec.resource_labels[k]: float(util[k]) for k in crit}
    crit_ok = all(abs(util[k] - 1.0) <= atol for k in crit)

    # non-bottleneck processing rows must be <= 1
    noncrit_proc = [k for k in spec.processing_resources if k not in crit]
    noncrit_ok = all(util[k] <= 1.0 + atol for k in noncrit_proc)

    # input rows: no-rejection fully used (= 1), rejection-allowed <= 1
    input_ok = True
    for k in spec.input_resources:
        block_use = float(np.sum(spec.resource_use[k] * beta))
        if k in spec.no_rejection_resources:
            input_ok = input_ok and abs(block_use - 1.0) <= atol
        else:
            input_ok = input_ok and block_use <= 1.0 + atol

    return {
        "flow_balance_residual": fb,
        "flow_balance_ok": bool(np.max(np.abs(fb)) <= atol),
        "utilization": util,
        "critical_load": crit_load,
        "critical_load_ok": bool(crit_ok),
        "noncritical_proc_ok": bool(noncrit_ok),
        "input_ok": bool(input_ok),
    }
