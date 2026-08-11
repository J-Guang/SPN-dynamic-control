"""Phase 0: lifted policy feasibility (math_foundation.md Section 5)."""
from __future__ import annotations

import numpy as np
import pytest

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.policies.allocation import allocation_rule
from diffusion_based_policy.policies.lifting import action_is_feasible, lifted_action


def _model(path):
    spec, params = load_network(path)
    return BCPModel(spec, params)


def _random_states(model, rng, count=200):
    """Integer queues including empty-buffer boundary states."""
    I = model.spec.num_buffers
    out = []
    for _ in range(count):
        q = rng.integers(0, 5, size=I).astype(float)
        # force some buffers empty
        mask = rng.random(I) < 0.4
        q[mask] = 0.0
        out.append(q)
    return out


@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_lifted_action_feasible(network_configs, name):
    model = _model(network_configs[name])
    rng = np.random.default_rng(0)
    for q in _random_states(model, rng):
        g = rng.standard_normal(model.spec.num_buffers)
        a = lifted_action(model, q, g)
        assert set(np.unique(a)).issubset({0.0, 1.0})
        assert action_is_feasible(model, q, a), f"{name}: infeasible a at q={q}"


@pytest.mark.parametrize("name", ["crisscross", "pesic_williams", "three_station_bigstep"])
def test_never_serves_empty_buffer(network_configs, name):
    model = _model(network_configs[name])
    spec = model.spec
    rng = np.random.default_rng(1)
    for q in _random_states(model, rng):
        g = rng.standard_normal(spec.num_buffers)
        a = lifted_action(model, q, g)
        for j in np.where(a > 0)[0]:
            for i in spec.consumed_buffers(j):
                assert q[i] >= spec.consumption[i, j] - 1e-9, (
                    f"{name}: activity {j} served with q[{i}]={q[i]}"
                )


def test_no_rejection_inputs_always_admitted(network_configs):
    """Crisscross / PW input rows (idle forbidden) must always be served."""
    for name in ("crisscross", "pesic_williams"):
        model = _model(network_configs[name])
        spec = model.spec
        rng = np.random.default_rng(2)
        for q in _random_states(model, rng, count=50):
            g = rng.standard_normal(spec.num_buffers)
            a = lifted_action(model, q, g)
            for k in spec.no_rejection_resources:
                used = float(spec.resource_use[k] @ a)
                nominal = float(spec.resource_use[k] @ model.params.nominal_allocation)
                assert abs(used - nominal) < 1e-9, f"{name}: input row {k} idled"


def test_bigstep_rejection_when_scores_negative(network_configs):
    """When all admission scores are negative, bigstep inputs reject (idle)."""
    model = _model(network_configs["three_station_bigstep"])
    spec = model.spec
    # large positive gradient => admitting is costly => reject all inputs
    g = np.ones(spec.num_buffers) * 10.0
    q = np.ones(spec.num_buffers) * 3.0
    a = lifted_action(model, q, g)
    for k in spec.input_resources:
        assert float(spec.resource_use[k] @ a) == 0.0, f"input {k} should reject"


def test_routing_block_selects_one(network_configs):
    """uA routing block selects at most one of a9, a10."""
    model = _model(network_configs["three_station_bigstep"])
    spec = model.spec
    rng = np.random.default_rng(3)
    uA = spec.resource_labels.index("uA")
    block = spec.resource_block(uA)
    for _ in range(100):
        g = rng.standard_normal(spec.num_buffers)
        q = rng.integers(1, 4, size=spec.num_buffers).astype(float)
        a = lifted_action(model, q, g)
        assert a[block].sum() <= 1.0 + 1e-9


def test_brownian_allocation_respects_empty_buffer(network_configs):
    """Allocation x never allocates to an activity consuming an empty buffer."""
    for name in ("crisscross", "pesic_williams", "three_station_bigstep"):
        model = _model(network_configs[name])
        spec = model.spec
        rng = np.random.default_rng(4)
        for _ in range(100):
            z = np.abs(rng.standard_normal(spec.num_buffers))
            z[rng.random(spec.num_buffers) < 0.4] = 0.0
            g = rng.standard_normal(spec.num_buffers)
            x = allocation_rule(model, z, g)
            for j in np.where(x > 0)[0]:
                for i in spec.consumed_buffers(j):
                    assert z[i] > 1e-12, f"{name}: x served empty buffer {i}"


def test_deterministic_tiebreak(network_configs):
    """Identical inputs give identical actions (deterministic policy)."""
    model = _model(network_configs["three_station_bigstep"])
    g = np.zeros(model.spec.num_buffers)  # ties everywhere
    q = np.ones(model.spec.num_buffers) * 2.0
    a1 = lifted_action(model, q, g)
    a2 = lifted_action(model, q, g)
    np.testing.assert_array_equal(a1, a2)
