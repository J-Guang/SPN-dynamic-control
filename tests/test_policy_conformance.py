"""Conformance B2: policy extraction matches the general formulation (3.5, 5.3-5.4).

Validates the block-decomposition policy/driver against the GENERAL definition:
the policy-index formula, and the LP over the feasible set solved independently
with scipy. Everything is recomputed from the config (R, K, c_tilde, A, beta), so
configs and the Q construction stay changeable -- only a wrong implementation
fails, not a changed value.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.policies.allocation import allocation_rule, hamiltonian_value
from diffusion_based_policy.policies.lifting import action_is_feasible, lifted_action_lp
from diffusion_based_policy.policies.vectorized import VectorizedPolicy

_RNG = np.random.default_rng(20260621)


def test_policy_index_formula(network_configs):
    """pi_j(z;g) = b (g . R_.j + (K^T c_tilde)_j)  (math_foundation 3.5)."""
    for name, path in network_configs.items():
        spec, p = load_network(path)
        m = BCPModel(spec, p)
        for _ in range(5):
            g = _RNG.standard_normal(spec.num_buffers)
            expected = p.b * (g @ m.R + m.K_matrix.T @ m.c_tilde)
            np.testing.assert_allclose(m.policy_index(g), expected, atol=1e-9, err_msg=name)


def _brownian_lp_optimum(model, z, g):
    """max sum_j pi_j x_j over X(z): 0<=x<=1, A x<=e, A_0 x=A_0 beta, (Cx)_i=0 if z_i=0."""
    spec = model.spec
    pi = model.policy_index(g)
    J = spec.num_activities
    beta = model.params.nominal_allocation

    A_eq, b_eq = [], []
    for k in spec.no_rejection_resources:
        A_eq.append(spec.resource_use[k])
        b_eq.append(float(spec.resource_use[k] @ beta))

    bounds = [(0.0, 1.0)] * J
    C = spec.consumption
    for i in np.where(np.asarray(z) <= 1e-12)[0]:
        for j in np.where(C[i] > 0)[0]:
            bounds[j] = (0.0, 0.0)

    res = linprog(
        -pi,
        A_ub=spec.resource_use,
        b_ub=np.ones(spec.num_resources),
        A_eq=np.asarray(A_eq) if A_eq else None,
        b_eq=np.asarray(b_eq) if A_eq else None,
        bounds=bounds,
        method="highs",
    )
    assert res.success
    return -res.fun


def test_brownian_allocation_is_lp_optimal(network_configs):
    """Block allocation_rule attains the general LP optimum over X(z) (3.5).

    Holds for any config whose processing rows are idle-admissible (all paper
    configs after the iota=Section-1.6 convention); validated for arbitrary g.
    """
    for name, path in network_configs.items():
        spec, p = load_network(path)
        m = BCPModel(spec, p)
        for _ in range(10):
            g = _RNG.standard_normal(spec.num_buffers)
            z = np.abs(_RNG.standard_normal(spec.num_buffers))
            z[_RNG.integers(spec.num_buffers)] = 0.0  # exercise a boundary face
            x = allocation_rule(m, z, g)
            pi = m.policy_index(g)
            assert float(pi @ x) == pytest.approx(_brownian_lp_optimum(m, z, g), abs=1e-6)


def test_hamiltonian_driver_matches_lp(network_configs):
    """Hamiltonian F(z,g) = h.z + min_x{...} matches the general LP for all networks
    (math_foundation 3.4 / 4.2).  The min term equals pi.beta - max_{x in X(z)} pi.x.
    """
    for name, path in network_configs.items():
        spec, p = load_network(path)
        m = BCPModel(spec, p)
        beta = p.nominal_allocation
        for _ in range(10):
            g = _RNG.standard_normal(spec.num_buffers)
            z = np.abs(_RNG.standard_normal(spec.num_buffers))
            z[_RNG.integers(spec.num_buffers)] = 0.0
            pi = m.policy_index(g)
            expected = float(m.h_tilde @ z + pi @ beta - _brownian_lp_optimum(m, z, g))
            assert hamiltonian_value(m, z, g) == pytest.approx(expected, abs=1e-6), name


def _action_from_selected(spec, selected):
    a = np.zeros(spec.num_activities)
    for j in selected:
        if j >= 0:
            a[int(j)] = 1.0
    return a


def test_vectorized_policy_matches_lp(network_configs):
    """The production vectorized lifted policy is feasible AND attains the general
    LP optimum over A(q) (math_foundation 5.3-5.4).

    This exercises the shared-buffer contention path too (Pesic-Williams b1), which
    the naive per-block argmax cannot resolve but the vectorized policy does.
    """
    for name, path in network_configs.items():
        spec, p = load_network(path)
        m = BCPModel(spec, p)
        vp = VectorizedPolicy(m)
        for _ in range(12):
            g = _RNG.standard_normal(spec.num_buffers)
            q = _RNG.integers(0, 4, size=spec.num_buffers).astype(float)
            pi = m.policy_index(g)
            sel = vp.select(q, g)[0]
            a = _action_from_selected(spec, sel)
            assert action_is_feasible(m, q, a), f"{name}: infeasible action at q={q}"
            a_lp = lifted_action_lp(m, q, g)
            assert float(pi @ a) == pytest.approx(float(pi @ a_lp), abs=1e-6), name
