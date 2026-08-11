"""Phase 0: configs load into NetworkSpec and pass shape/sign validation."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from diffusion_based_policy.bcp import BCPModel, bcp_params_from_dict
from diffusion_based_policy.config import load_network
from diffusion_based_policy.network import NetworkSpec, network_from_dict
from diffusion_based_policy.validation import validate_network_spec, validate_supported_scope

EXPECTED_SIZES = {
    "crisscross": (3, 5, 4),
    "pesic_williams": (3, 8, 6),
    "three_station_bigstep": (8, 12, 6),
}


def test_all_configs_load(network_configs):
    for name, path in network_configs.items():
        spec, params = load_network(path)
        assert isinstance(spec, NetworkSpec)
        I, J, K = EXPECTED_SIZES[name]
        assert (spec.num_buffers, spec.num_activities, spec.num_resources) == (I, J, K)
        assert spec.buffer_change.shape == (I, J)
        assert spec.resource_use.shape == (K, J)
        assert params.critical_rates.shape == (J,)
        assert params.nominal_allocation.shape == (J,)


def test_validation_clean(network_configs):
    for name, path in network_configs.items():
        spec, _ = load_network(path)
        msgs = validate_network_spec(spec)
        assert msgs == [], f"{name}: {msgs}"


def test_consumption_matrix(network_configs):
    spec, _ = load_network(network_configs["crisscross"])
    # C = (-P)^+; for crisscross s1 (col 2) consumes buffer 1.
    C = spec.consumption
    assert C.shape == spec.buffer_change.shape
    assert np.all(C >= 0)
    assert C[0, 2] == 1.0   # s1 consumes b1
    # input activities consume nothing
    a1 = spec.activity_labels.index("a1")
    assert np.all(C[:, a1] == 0)


def test_resource_metadata(network_configs):
    spec_cc, _ = load_network(network_configs["crisscross"])
    # Criss-Cross (math_foundation 1.6): inputs no-rejection, processing idle-admissible.
    assert spec_cc.idle_allowed.tolist() == [False, False, True, True]

    spec, _ = load_network(network_configs["pesic_williams"])
    # PW: S1,S2,S3 processing (idle allowed), u1,u2,u3 input (no rejection)
    assert spec.is_processing.sum() == 3
    assert spec.is_input.sum() == 3
    assert set(spec.no_rejection_resources.tolist()) == {3, 4, 5}

    spec_big, _ = load_network(network_configs["three_station_bigstep"])
    # bigstep: all inputs allow rejection => no no-rejection rows
    assert spec_big.no_rejection_resources.size == 0


def test_resource_blocks(network_configs):
    spec, _ = load_network(network_configs["three_station_bigstep"])
    blocks = spec.resource_blocks
    # S3 serves s3, s4, s6, s8 (indices 2,3,5,7)
    s3_block = blocks[spec.resource_labels.index("S3")]
    assert set(s3_block.tolist()) == {2, 3, 5, 7}
    # uA routes a9, a10 (indices 8, 9)
    uA_block = blocks[spec.resource_labels.index("uA")]
    assert set(uA_block.tolist()) == {8, 9}


def test_supported_scope_passes_for_paper_networks(network_configs):
    for name, path in network_configs.items():
        spec, params = load_network(path)
        model = BCPModel(spec, params)
        assert validate_supported_scope(model) == [], name


def _raw(path):
    with open(path) as f:
        return yaml.safe_load(f)


def test_supported_scope_flags_multi_resource_activity(network_configs):
    d = _raw(network_configs["crisscross"])
    # make activity s1 (col 2) also use resource S2 (row 3) -> multi-resource
    d = copy.deepcopy(d)
    d["resource_use"][3][2] = 1.0
    spec = network_from_dict(d)
    model = BCPModel(spec, bcp_params_from_dict(d["bcp"], spec))
    msgs = validate_supported_scope(model)
    assert any("multiple resources" in m for m in msgs), msgs


def test_block_policy_requires_single_resource_per_activity(network_configs):
    """The block hot path (VectorizedPolicy) fails fast with NotImplementedError on a
    multi-resource activity, instead of silently giving a wrong block-argmax."""
    from diffusion_based_policy.policies.vectorized import VectorizedPolicy

    # in-scope paper network: builds fine
    d0 = _raw(network_configs["crisscross"])
    spec0 = network_from_dict(d0)
    VectorizedPolicy(BCPModel(spec0, bcp_params_from_dict(d0["bcp"], spec0)))  # no raise

    # out-of-scope: make s1 (col 2) also use S2 (row 3) -> multi-resource activity
    d = copy.deepcopy(d0)
    d["resource_use"][3][2] = 1.0
    spec = network_from_dict(d)
    model = BCPModel(spec, bcp_params_from_dict(d["bcp"], spec))
    with pytest.raises(NotImplementedError):
        VectorizedPolicy(model)


def test_supported_scope_flags_multi_activity_no_rejection(network_configs):
    # Pesic-Williams input u1 is no-rejection single-activity; add a second basic
    # activity to its block to violate the single-activity assumption.
    d = copy.deepcopy(_raw(network_configs["pesic_williams"]))
    # u1 is resource row index 3; activity a2 (index 6) -> also serve u1
    d["resource_use"][3][6] = 1.0
    spec = network_from_dict(d)
    model = BCPModel(spec, bcp_params_from_dict(d["bcp"], spec))
    msgs = validate_supported_scope(model)
    assert any("no-rejection" in m or "multiple resources" in m for m in msgs), msgs
