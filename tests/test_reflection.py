"""Phase 1: reflection backends solve the LCP and agree with each other."""
from __future__ import annotations

import os

import numpy as np
import pytest

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.reflection import available_backends, get_solver, lcp_residual

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _H(network_configs, name):
    spec, params = load_network(network_configs[name])
    return BCPModel(spec, params).H


# --------------------------------------------------------------------- numba
@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_numba_lcp_conditions(network_configs, name):
    H = _H(network_configs, name)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((1000, H.shape[0])) * 0.5
    res = get_solver("numba", H).solve(X)
    Y, L = res.reflected, res.boundary_push
    assert np.all(Y >= -1e-8)
    assert np.all(L >= -1e-8)
    # complementarity Y_i L_i = 0
    assert np.max(np.abs(Y * L)) < 1e-6
    # Y = X + H L
    assert np.max(np.abs(Y - (X + L @ H.T))) < 1e-6
    assert res.lcp_residual < 1e-6


def test_positive_passthrough(network_configs):
    H = _H(network_configs, "three_station_bigstep")
    rng = np.random.default_rng(1)
    X = np.abs(rng.standard_normal((200, H.shape[0]))) + 0.1
    Y = get_solver("numba", H).project(X)
    np.testing.assert_allclose(Y, X, atol=1e-10)


def test_known_2d_case():
    # Legacy regression case: H upper-triangular, both components negative.
    H = np.array([[0.25, 0.2475], [0.0, 0.25]])
    Y = get_solver("numba", H).project(np.array([[-0.1, -0.5]]))
    np.testing.assert_allclose(Y, [[0.395, 0.0]], atol=1e-6)


def test_bigstep_project_uses_vectorized_basis_enum(network_configs):
    H = _H(network_configs, "three_station_bigstep")
    solver = get_solver("numba", H)
    assert solver._enum_vec is not None

    B_mask, _A, _HA, _valid = solver._enum_vec
    assert B_mask.shape == (1 << H.shape[0], H.shape[0])

    rng = np.random.default_rng(123)
    X = -np.abs(rng.standard_normal((512, H.shape[0]))) * 0.2
    Y = solver.project(X)
    assert lcp_residual(X, Y, H) < 1e-6


def test_m_matrix_backend_accepts_m_matrix():
    H = np.array([[1.0, -0.2], [-0.1, 1.0]])
    X = np.array([[-0.5, -0.25], [0.1, -0.3]])
    res = get_solver("m_matrix", H).solve(X)
    assert np.all(res.reflected >= -1e-8)
    assert res.lcp_residual < 1e-6


# --------------------------------------------------------------------- Lemke
@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_lemke_lcp_conditions(network_configs, name):
    H = _H(network_configs, name)
    rng = np.random.default_rng(17)
    X = rng.standard_normal((256, H.shape[0])) * 0.5
    res = get_solver("lemke", H).solve(X)
    Y, L = res.reflected, res.boundary_push
    assert np.all(Y >= -1e-8)
    assert np.all(L >= -1e-8)
    assert np.max(np.abs(Y * L)) < 1e-6
    assert np.max(np.abs(Y - (X + L @ H.T))) < 1e-6
    assert res.lcp_residual < 1e-6


def test_lemke_matches_numba_bigstep_boundary_batch(network_configs):
    H = _H(network_configs, "three_station_bigstep")
    rng = np.random.default_rng(1234)
    X = -np.abs(rng.standard_normal((512, H.shape[0]))) * 0.2
    y_lemke = get_solver("lemke", H).project(X)
    y_numba = get_solver("numba", H).project(X)
    assert lcp_residual(X, y_lemke, H) < 1e-6
    np.testing.assert_allclose(y_lemke, y_numba, atol=1e-6)


def test_lemke_scaling_is_positive_and_equivalent(network_configs):
    H = _H(network_configs, "three_station_bigstep")
    solver = get_solver("lemke", H)
    assert np.all(solver.row_scale > 0.0)
    assert np.all(solver.col_scale > 0.0)
    np.testing.assert_allclose(
        solver.M_scaled,
        solver.row_scale[:, None] * H * solver.col_scale[None, :],
        rtol=1e-12,
        atol=1e-12,
    )


# --------------------------------------------------------- legacy fixtures
@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_matches_legacy_fixture(name):
    path = os.path.join(FIX, f"reflection_{name}.npz")
    if not os.path.exists(path):
        pytest.skip("fixtures not generated; run cli/make_fixtures.py")
    data = np.load(path)
    X, Y_ref, H = data["X"], data["Y"], data["H"]
    res = get_solver("numba", H).solve(X)
    # our backend should solve the LCP at least as accurately as the legacy ref
    assert res.lcp_residual < 1e-7
    # and agree with the legacy solution (both solve the same LCP)
    assert np.max(np.abs(res.reflected - Y_ref)) < 1e-4


# ------------------------------------------------------------- CPU vs GPU
@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_cpu_gpu_agreement(network_configs, name):
    if "gpu" not in available_backends():
        pytest.skip("tensorflow not importable; gpu backend unavailable")
    H = _H(network_configs, name)
    rng = np.random.default_rng(11)
    X = rng.standard_normal((256, H.shape[0])) * 0.5
    y_cpu = get_solver("numba", H).project(X)
    y_gpu = get_solver("gpu", H).project(X)
    assert np.max(np.abs(y_cpu - y_gpu)) < 1e-6
    # GPU result also satisfies the LCP
    assert lcp_residual(X, y_gpu, H) < 1e-6


def test_gpu_boundary_push(network_configs):
    if "gpu" not in available_backends():
        pytest.skip("tensorflow not importable; gpu backend unavailable")
    H = _H(network_configs, "crisscross")
    rng = np.random.default_rng(3)
    X = rng.standard_normal((64, H.shape[0])) * 0.6
    res = get_solver("gpu", H).solve(X)
    # Y = X + H L reconstructs
    assert np.max(np.abs(res.reflected - (X + res.boundary_push @ H.T))) < 1e-6
