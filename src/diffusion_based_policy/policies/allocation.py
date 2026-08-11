"""Brownian allocation rule (math_foundation.md Sections 3.5, 3.6, 4.2).

Given a gradient g at a diffusion state z, choose

    x*(z) in argmax_{x in X(z)} sum_j pi_j(z; g) x_j,

which for the single-resource paper examples decomposes by resource block.
This module also exposes the Hamiltonian min-term used by the BSDE driver F.
"""
from __future__ import annotations

import numpy as np

from ..bcp import BCPModel

_TOL = 1e-12


def _empty_buffer_mask(spec, z: np.ndarray) -> np.ndarray:
    """Activities forbidden because they consume a buffer with z_i <= 0."""
    empty = z <= _TOL                       # (I,)
    # activity j forbidden if any consumed buffer is empty
    forbidden = (spec.consumption[empty] > 0).any(axis=0) if empty.any() else \
        np.zeros(spec.num_activities, dtype=bool)
    return forbidden


def block_scores(model: BCPModel, g: np.ndarray) -> np.ndarray:
    """pi_j(z; g) for all activities (Sec 3.5)."""
    return model.policy_index(g)


def allocation_rule(model: BCPModel, z: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Return the bang-bang allocation x in {0,1}^J solving the block LP at z.

    No-rejection input rows are fixed to beta (A_0 x = A_0 beta).
    """
    spec = model.spec
    pi = model.policy_index(g)
    forbidden = _empty_buffer_mask(spec, np.asarray(z, dtype=np.float64))
    x = np.zeros(spec.num_activities)

    for k in range(spec.num_resources):
        block = spec.resource_block(k)
        if block.size == 0:
            continue
        if k in spec.no_rejection_resources:
            # fixed allocation A_0 x = A_0 beta: serve the nominal input activity
            x[block] = model.params.nominal_allocation[block]
            continue
        feasible = block[~forbidden[block]]
        if feasible.size == 0:
            continue  # idle
        j_star = feasible[int(np.argmax(pi[feasible]))]
        idle_allowed = bool(spec.idle_allowed[k])
        if (not idle_allowed) or pi[j_star] > 0:
            x[j_star] = 1.0
    return x


def hamiltonian_value(model: BCPModel, z: np.ndarray, g: np.ndarray) -> float:
    """Driver F(z, g) = htilde.z + min_x { g.R theta_x + c.K theta_x }.

    Using theta_x = b(beta - x), the min term equals
        sum_j beta_j pi_j  -  max_x sum_j pi_j x_j
    over the optimized blocks (no-rejection inputs cancel).
    """
    spec = model.spec
    pi = model.policy_index(g)
    beta = model.params.nominal_allocation
    z = np.asarray(z, dtype=np.float64)
    forbidden = _empty_buffer_mask(spec, z)

    min_term = 0.0
    for k in range(spec.num_resources):
        if k in spec.no_rejection_resources:
            continue  # fixed; contributes 0 net to the min term
        block = spec.resource_block(k)
        if block.size == 0:
            continue
        feasible = block[~forbidden[block]]
        block_max = 0.0
        if feasible.size:
            best = float(np.max(pi[feasible]))
            if bool(spec.idle_allowed[k]):
                block_max = max(0.0, best)
            else:
                block_max = best
        min_term += float(np.sum(beta[block] * pi[block])) - block_max

    return float(model.h_tilde @ z + min_term)


def hamiltonian_min_term_batch(model: BCPModel, z: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Vectorized numpy version of the Hamiltonian min-term over a batch.

    z, g: (B, I).  Returns (B,) min-term values (without the htilde.z piece).
    Used by reference numpy checks; the TF training path has its own copy.
    """
    spec = model.spec
    z = np.atleast_2d(np.asarray(z, dtype=np.float64))
    g = np.atleast_2d(np.asarray(g, dtype=np.float64))
    pi = model.policy_index(g)                     # (B, J)
    beta = model.params.nominal_allocation
    B = z.shape[0]
    empty = z <= _TOL                              # (B, I)
    # forbidden[b, j] = any consumed buffer empty
    C = spec.consumption                           # (I, J)
    forbidden = np.einsum("bi,ij->bj", empty.astype(np.float64), (C > 0).astype(np.float64)) > 0

    out = np.zeros(B)
    for k in range(spec.num_resources):
        if k in spec.no_rejection_resources:
            continue
        block = spec.resource_block(k)
        if block.size == 0:
            continue
        pib = pi[:, block]                         # (B, |block|)
        fb = forbidden[:, block]                   # (B, |block|)
        masked = np.where(fb, -np.inf, pib)
        best = np.max(masked, axis=1)
        best = np.where(np.isfinite(best), best, 0.0)
        if bool(spec.idle_allowed[k]):
            block_max = np.maximum(0.0, best)
        else:
            block_max = best
        out += pib @ beta[block] - block_max
    return out
