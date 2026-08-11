"""Lifted prelimit policy (math_foundation.md Section 5).

Map an integer queue q to an executable action a in {0,1}^J that maximizes the
BCP index over the executable feasible set

    A(q) = { a in {0,1}^J : A a <= e, C a <= q, A_0 a = A_0 beta }.

For the paper examples this decomposes by resource block; a general LP fallback
guarantees feasibility for any network.
"""
from __future__ import annotations

import numpy as np

from ..bcp import BCPModel

_TOL = 1e-9


def _scores_from_q(model: BCPModel, q: np.ndarray, g: np.ndarray | None) -> np.ndarray:
    if g is None:
        z = np.asarray(q, dtype=np.float64) / np.sqrt(model.params.n)
        g = z  # placeholder; callers normally pass g explicitly
    return model.policy_index(np.asarray(g, dtype=np.float64))


def _feasible_in_block(spec, block: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Activities j in block with C_.j <= q (enough jobs in every consumed buffer)."""
    C = spec.consumption
    ok = []
    for j in block:
        if np.all(C[:, j] <= q + _TOL):
            ok.append(j)
    return np.asarray(ok, dtype=int)


def lifted_action_block(model: BCPModel, q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Block-decomposed lifted action (Sec 5.4). Deterministic tie-break = lowest index."""
    spec = model.spec
    q = np.asarray(q, dtype=np.float64)
    pi = model.policy_index(np.asarray(g, dtype=np.float64))
    a = np.zeros(spec.num_activities)

    for k in range(spec.num_resources):
        block = spec.resource_block(k)
        if block.size == 0:
            continue
        if k in spec.no_rejection_resources:
            # always admit nominal input activities (A_0 a = A_0 beta)
            for j in block:
                if model.params.nominal_allocation[j] > 0:
                    a[j] = 1.0
            continue
        feasible = _feasible_in_block(spec, block, q)
        if feasible.size == 0:
            continue  # idle / reject
        j_star = int(feasible[int(np.argmax(pi[feasible]))])
        idle_allowed = bool(spec.idle_allowed[k])
        if (not idle_allowed) or pi[j_star] > 0:
            a[j_star] = 1.0
    return a


def action_is_feasible(model: BCPModel, q: np.ndarray, a: np.ndarray) -> bool:
    spec = model.spec
    q = np.asarray(q, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if np.any(a < -_TOL) or np.any(a > 1 + _TOL):
        return False
    if np.any(spec.resource_use @ a > 1 + _TOL):
        return False
    if np.any(spec.consumption @ a > q + _TOL):
        return False
    for k in spec.no_rejection_resources:
        lhs = float(spec.resource_use[k] @ a)
        rhs = float(spec.resource_use[k] @ model.params.nominal_allocation)
        if abs(lhs - rhs) > 1e-6:
            return False
    return True


def lifted_action_lp(model: BCPModel, q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """General LP/MILP fallback over A(q)."""
    from scipy.optimize import linprog

    spec = model.spec
    q = np.asarray(q, dtype=np.float64)
    pi = model.policy_index(np.asarray(g, dtype=np.float64))
    J = spec.num_activities

    A_ub = np.vstack([spec.resource_use, spec.consumption])
    b_ub = np.concatenate([np.ones(spec.num_resources), q])

    A_eq, b_eq = [], []
    for k in spec.no_rejection_resources:
        A_eq.append(spec.resource_use[k])
        b_eq.append(float(spec.resource_use[k] @ model.params.nominal_allocation))

    res = linprog(
        -pi,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=np.asarray(A_eq) if A_eq else None,
        b_eq=np.asarray(b_eq) if b_eq else None,
        bounds=[(0, 1)] * J,
        integrality=np.ones(J),
        method="highs",
    )
    if not res.success:
        return np.zeros(J)
    return np.round(res.x)


def lifted_action(model: BCPModel, q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Lifted action with block fast-path and LP fallback if block result is
    infeasible for the executable set (should not happen for the paper examples)."""
    a = lifted_action_block(model, q, g)
    if action_is_feasible(model, q, a):
        return a
    return lifted_action_lp(model, q, g)


def lifted_action_batch(model: BCPModel, q: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Vectorized block lifting over a batch of (q, g). Shapes (B, I_or_J)."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    g = np.atleast_2d(np.asarray(g, dtype=np.float64))
    B = q.shape[0]
    out = np.zeros((B, model.spec.num_activities))
    for b in range(B):
        out[b] = lifted_action(model, q[b], g[b])
    return out
