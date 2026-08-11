"""Conformance B1: heavy-traffic CONSISTENCY invariants (math_foundation.md 1.5, 2.1-2.3).

These assert *relationships* that must hold for ANY valid heavy-traffic config,
recomputed from the config itself. They deliberately do NOT freeze the paper's
specific numbers, so configs stay freely changeable -- only an internally
inconsistent config fails. This is the permanent version of the ad-hoc
config<->doc audit.
"""
from __future__ import annotations

import numpy as np

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network


def test_flow_balance(network_configs):
    """P diag(mu*) beta = 0  (math_foundation 1.5 / 2.1)."""
    for name, path in network_configs.items():
        spec, p = load_network(path)
        flow = spec.buffer_change @ (p.critical_rates * p.nominal_allocation)
        np.testing.assert_allclose(flow, 0.0, atol=1e-9, err_msg=name)


def test_critical_loading(network_configs):
    """Some processing station is a bottleneck (A beta = 1), none is overloaded, and the
    DERIVED critical set = { processing k : (A beta)_k = 1 } (math_foundation 2.1)."""
    for name, path in network_configs.items():
        spec, p = load_network(path)
        rho = spec.resource_use @ p.nominal_allocation
        proc = [k for k in range(spec.num_resources) if spec.is_processing[k]]
        assert all(rho[k] <= 1.0 + 1e-9 for k in proc), f"{name}: processing overloaded"
        crit = {k for k in proc if abs(rho[k] - 1.0) < 1e-9}
        assert crit, f"{name}: no critical processing station (A beta = 1)"
        assert {int(k) for k in p.critical_resources} == crit, f"{name}: critical mismatch"


def test_perturbation_consistency(network_configs):
    """mu_hat = sqrt(n) (mu^(n) - mu*)  links rates, mu*, mu_hat, n (2.1)."""
    for name, path in network_configs.items():
        spec, p = load_network(path)
        expected = np.sqrt(p.n) * (spec.rates - p.critical_rates)
        np.testing.assert_allclose(p.rate_perturbation, expected, atol=1e-9, err_msg=name)


def test_discount_consistency(network_configs):
    """gamma = n rho^(n)  (2.2): the nondegenerate discounted Brownian limit."""
    for name, path in network_configs.items():
        spec, p = load_network(path)
        assert abs(p.gamma - p.n * spec.discount) < 1e-9, name


def test_nonbasic_matches_zero_beta(network_configs):
    """Nonbasic activities are exactly { j : beta_j = 0 } (math_foundation 3.1)."""
    for name, path in network_configs.items():
        spec, p = load_network(path)
        derived = {int(j) for j in np.where(np.abs(p.nominal_allocation) < 1e-12)[0]}
        assert {int(j) for j in p.nonbasic_activities} == derived, name


def test_cost_scaling_consistency(network_configs):
    """Brownian input cost c_tilde_k = mu_bar_k * (rejection_cost_k / n)  (2.3).

    rejection_cost is the prelimit per-job charge c^{D,(n)}; bar c^D = c^{D,(n)}/n.
    """
    for name, path in network_configs.items():
        spec, p = load_network(path)
        m = BCPModel(spec, p)
        mu_bar = m.input_opportunity_rate
        for (kind, idx), val in zip(m._K_rows, m.c_tilde):  # noqa: SLF001
            if kind == "resource" and spec.is_input[idx]:
                expected = mu_bar[idx] * spec.rejection_cost[idx] / p.n
                assert abs(val - expected) < 1e-12, f"{name} input row {idx}"
