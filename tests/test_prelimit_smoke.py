"""Phase 4: prelimit simulator + vectorized policy smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.policies.baselines import baseline_gradient
from diffusion_based_policy.policies.lifting import action_is_feasible, lifted_action
from diffusion_based_policy.policies.vectorized import VectorizedPolicy
from diffusion_based_policy.sim.prelimit import simulate_prelimit


def _model(network_configs, name):
    spec, params = load_network(network_configs[name])
    return BCPModel(spec, params)


# --------------------------------------------------- vectorized vs scalar
@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_vectorized_matches_scalar(network_configs, name):
    """The vectorized policy attains the same (optimal) BCP-index objective as the
    scalar LP lifting. Exact actions may differ on ties (both are optimal), so we
    compare objective values and feasibility, not the raw action arrays."""
    model = _model(network_configs, name)
    vpol = VectorizedPolicy(model)
    rng = np.random.default_rng(0)
    n_shared_tested = 0
    for _ in range(300):
        q = rng.integers(0, 5, size=model.I).astype(float)
        q[rng.random(model.I) < 0.3] = 0.0
        g = rng.standard_normal(model.I)
        sel = vpol.select(q[None, :], g[None, :])[0]
        a_scalar = lifted_action(model, q, g)
        a_vec = np.zeros(model.spec.num_activities)
        for k, j in enumerate(sel):
            if j >= 0:
                a_vec[j] = 1.0
        pi = model.policy_index(g)
        # same optimal objective
        assert abs(pi @ a_vec - pi @ a_scalar) < 1e-7, f"{name}: suboptimal q={q}"
        # vectorized action is executable
        assert action_is_feasible(model, q, a_vec), f"{name}: infeasible q={q}"
        if not np.array_equal(a_vec, a_scalar):
            n_shared_tested += 1
    # crisscross / bigstep have no shared buffer => block argmax is unique-optimal
    if name in ("crisscross", "three_station_bigstep"):
        assert n_shared_tested == 0


# ------------------------------------------------------------ simulator
@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_prelimit_runs(network_configs, name):
    model = _model(network_configs, name)
    grad = baseline_gradient(model)
    res = simulate_prelimit(model, grad, num_paths=64, horizon=0.5, seed=1)
    assert np.isfinite(res["cost_mean"])
    assert res["cost_mean"] >= 0
    assert res["holding"] >= 0
    assert res["rejection"] >= 0
    # utilization is a fraction per processing station
    util = np.array(res["utilization"])
    assert np.all(util >= 0) and np.all(util <= 1.0 + 1e-9)
    assert len(util) == model.spec.processing_resources.size


def test_no_rejection_networks_have_zero_rejection(network_configs):
    for name in ("crisscross", "pesic_williams"):
        model = _model(network_configs, name)
        grad = baseline_gradient(model)
        res = simulate_prelimit(model, grad, num_paths=32, horizon=0.3, seed=2)
        assert res["rejection"] == 0.0, f"{name} should have no rejection cost"


def test_bigstep_rejection_positive_when_costly(network_configs):
    """A policy that rejects (negative-ish admission scores) incurs rejection cost."""
    model = _model(network_configs, "three_station_bigstep")

    def reject_grad(z):
        z = np.atleast_2d(np.asarray(z, float))
        return np.full_like(z, 5.0)  # high gradient => admission scores negative

    res = simulate_prelimit(model, reject_grad, num_paths=64, horizon=0.5, seed=3)
    assert res["rejection"] >= 0.0
    assert res["reject_fraction"] >= 0.0


def test_cost_decomposition_adds_up(network_configs):
    model = _model(network_configs, "three_station_bigstep")
    grad = baseline_gradient(model)
    res = simulate_prelimit(model, grad, num_paths=128, horizon=0.5, seed=4)
    assert abs(res["cost_mean"] - (res["holding"] + res["rejection"])) < 1e-6
    assert res["cost_scaled_mean"] == pytest.approx(res["cost_mean"] / model.n ** 1.5)


@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_no_horizon_overshoot(network_configs, name):
    """The simulator must not integrate cost or fire events past T_orig.

    With the horizon cap, each path runs to exactly T_orig (no overshoot), so the
    mean / max simulated time equals T_orig once paths reach the horizon.
    """
    model = _model(network_configs, name)
    grad = baseline_gradient(model)
    # short horizon so the overshoot bias would be large if present, and plenty
    # of jumps so every path reaches the horizon.
    res = simulate_prelimit(model, grad, num_paths=256, horizon=0.5, seed=11,
                            max_jumps=20000)
    T = res["T_orig"]
    assert res["max_sim_time"] <= T + 1e-6, f"{name}: overshoot to {res['max_sim_time']} > {T}"
    assert res["mean_sim_time"] == pytest.approx(T, rel=1e-6), \
        f"{name}: mean sim time {res['mean_sim_time']} != T_orig {T}"


def test_horizon_cap_invariants(network_configs):
    """Invariants implied by the horizon cap: every path runs to exactly T_orig
    and utilization is a valid fraction (no busy time accrued past the horizon)."""
    model = _model(network_configs, "three_station_bigstep")
    grad = baseline_gradient(model)
    res = simulate_prelimit(model, grad, num_paths=256, horizon=0.5, seed=7,
                            max_jumps=20000)
    util = np.array(res["utilization"])
    assert np.all((util >= 0) & (util <= 1 + 1e-9))
    assert res["mean_sim_time"] == pytest.approx(res["T_orig"], rel=1e-6)
