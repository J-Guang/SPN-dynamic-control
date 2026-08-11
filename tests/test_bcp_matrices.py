"""Phase 0: BCP matrices match math_foundation.md exactly.

Ground-truth values (zeta, Gamma, H, Q, c_tilde) are transcribed from
math_foundation.md Sections 2.4 and 3.6 with omega = 0.99.
"""
from __future__ import annotations

import numpy as np
import pytest

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.costs import verify_rejection_scaling
from diffusion_based_policy.validation import validate_bcp

W = 0.99  # omega in all three examples


def _model(path):
    spec, params = load_network(path)
    return BCPModel(spec, params)


# --- expected zeta / Gamma ---------------------------------------------------
EXPECTED_ZETA = {
    "crisscross": np.array([0.0, -1.0, 0.0]),
    "pesic_williams": np.array([-1.0, -1.0, -1.0]),
    "three_station_bigstep": np.zeros(8),
}

EXPECTED_GAMMA = {
    "crisscross": np.array([[2.0, 0.0, 0.0], [0.0, 2.0, -1.0], [0.0, -1.0, 2.0]]),
    "pesic_williams": np.diag([4.0, 2.0, 2.0]),
}


def _bigstep_gamma():
    G = np.diag([0.5] * 8)
    for i, j in [(0, 1), (3, 4), (5, 6), (6, 7)]:
        G[i, j] = G[j, i] = -0.25
    return G


EXPECTED_GAMMA["three_station_bigstep"] = _bigstep_gamma()


# --- expected H --------------------------------------------------------------
EXPECTED_H = {
    "crisscross": np.array([
        [1.0, -W, 0.0],
        [-W, 1.0, 0.0],
        [W, -1.0, 1.0],
    ]),
    "pesic_williams": np.array([
        [2.0, -W, 0.0],
        [-W, 1.0, 0.0],
        [-W, 0.0, 1.0],
    ]),
    "three_station_bigstep": np.array([
        [1/4, 0, 0, 0, 0, 0, -W/4, 0],
        [-1/4, 1/4, 0, 0, -W/12, 0, W/4, 0],
        [0, 0, 1/4, -W/12, 0, -W/12, 0, -W/12],
        [0, 0, -W/12, 1/4, 0, -W/12, 0, -W/12],
        [0, -3*W/4, W/12, -1/4, 1/4, W/12, 0, W/12],
        [0, 0, -W/12, -W/12, 0, 1/4, 0, -W/12],
        [-W/4, 0, W/12, W/12, 0, -1/4, 1/4, W/12],
        [W/4, 0, -W/12, -W/12, 0, -W/12, -1/4, 1/4],
    ]),
}


def test_zeta(network_configs):
    for name, path in network_configs.items():
        m = _model(path)
        np.testing.assert_allclose(m.zeta, EXPECTED_ZETA[name], atol=1e-12, err_msg=name)


def test_gamma(network_configs):
    for name, path in network_configs.items():
        m = _model(path)
        np.testing.assert_allclose(m.Gamma, EXPECTED_GAMMA[name], atol=1e-12, err_msg=name)


def test_sigma_factorization(network_configs):
    for name, path in network_configs.items():
        m = _model(path)
        np.testing.assert_allclose(m.sigma @ m.sigma.T, m.Gamma, atol=1e-10, err_msg=name)


def test_R_equals_minus_P_diag_mustar(network_configs):
    for name, path in network_configs.items():
        spec, params = load_network(path)
        m = BCPModel(spec, params)
        expected = -spec.buffer_change * params.critical_rates[None, :]
        np.testing.assert_allclose(m.R, expected, atol=1e-12, err_msg=name)


def test_H_equals_RQ(network_configs):
    for name, path in network_configs.items():
        m = _model(path)
        np.testing.assert_allclose(m.H, m.R @ m.Q, atol=1e-12, err_msg=name)


def test_H_matches_document(network_configs):
    for name, path in network_configs.items():
        m = _model(path)
        np.testing.assert_allclose(m.H, EXPECTED_H[name], atol=1e-12, err_msg=name)


def test_KQ_nonnegative(network_configs):
    for name, path in network_configs.items():
        m = _model(path)
        KQ = m.K_matrix @ m.Q
        assert np.min(KQ) >= -1e-12, f"{name}: min(KQ)={np.min(KQ)}"


def test_A0_Q_zero(network_configs):
    """No-rejection input rows: A_k Q = 0 (boundary preserves input equality)."""
    for name, path in network_configs.items():
        spec, params = load_network(path)
        m = BCPModel(spec, params)
        for k in spec.no_rejection_resources:
            row = spec.resource_use[k] @ m.Q
            assert np.max(np.abs(row)) < 1e-12, f"{name} row {k}: {row}"


def test_chi_is_zero(network_configs):
    """For all three paper examples Q is chosen so the boundary cost chi = 0."""
    for name, path in network_configs.items():
        m = _model(path)
        np.testing.assert_allclose(m.chi, 0.0, atol=1e-12, err_msg=name)


def test_c_tilde_bigstep(network_configs):
    """Input rows carry mu_bar_k * c^D_k / n with the per-job cost taken from the
    config (robust to the rejection-cost level); processing rows are zero."""
    m = _model(network_configs["three_station_bigstep"])
    spec, n = m.spec, m.params.n
    arr = {"uA": 0.5, "uB": 0.25, "uC": 0.25}   # structural input arrival rates
    got = {c["resource"]: c["c_tilde"] for c in verify_rejection_scaling(m)["checks"]}
    for r, rate in arr.items():
        cD = spec.rejection_cost[spec.resource_labels.index(r)]
        assert abs(got[r] - rate * cD / n) < 1e-12, r


def test_rejection_scaling(network_configs):
    """tilde c_k = mu_bar_k c^D_k / n on input rows (math_foundation 2.3)."""
    for name, path in network_configs.items():
        m = _model(path)
        res = verify_rejection_scaling(m)
        assert res["ok"], f"{name}: {res}"
    # bigstep charges a positive rejection cost on all three input streams
    m = _model(network_configs["three_station_bigstep"])
    checks = verify_rejection_scaling(m)["checks"]
    assert {c["resource"] for c in checks} == {"uA", "uB", "uC"}
    assert all(c["c_tilde"] > 0 for c in checks)


def test_policy_index_offset_bigstep(network_configs):
    """(K^T c_tilde)_j carries the input resource's c_tilde on its admission
    activities and zero on processing activities."""
    spec, params = load_network(network_configs["three_station_bigstep"])
    m = BCPModel(spec, params)
    off = m.index_offset
    by_res = {c["resource"]: c["c_tilde"] for c in verify_rejection_scaling(m)["checks"]}
    a9 = spec.activity_labels.index("a9")    # uA admission
    a11 = spec.activity_labels.index("a11")  # uB admission
    s1 = spec.activity_labels.index("s1")
    assert abs(off[a9] - by_res["uA"]) < 1e-12
    assert abs(off[a11] - by_res["uB"]) < 1e-12
    assert abs(off[s1]) < 1e-12              # processing rows have zero cost


def test_bcp_validation_clean(network_configs):
    for name, path in network_configs.items():
        m = _model(path)
        assert validate_bcp(m) == [], name
