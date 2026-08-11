"""Phase 0: static planning and critical-pair verification."""
from __future__ import annotations

import numpy as np

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.planning import (
    flow_balance_residual,
    static_planning_lp,
    verify_critical_pair,
)


def _model(path):
    spec, params = load_network(path)
    return spec, params, BCPModel(spec, params)


def test_flow_balance_zero(network_configs):
    """P diag(mu*) beta = 0 for all three networks (math_foundation 2.1)."""
    for name, path in network_configs.items():
        spec, params, _ = _model(path)
        res = flow_balance_residual(
            spec.buffer_change, params.critical_rates, params.nominal_allocation
        )
        assert np.max(np.abs(res)) < 1e-9, f"{name}: {res}"


def test_critical_load(network_configs):
    """A beta = 1 on every listed critical processing resource."""
    for name, path in network_configs.items():
        spec, params, model = _model(path)
        v = verify_critical_pair(model)
        assert v["flow_balance_ok"], name
        assert v["critical_load_ok"], f"{name}: {v['critical_load']}"
        assert v["noncritical_proc_ok"], name
        assert v["input_ok"], name


def test_critical_rho_is_one(network_configs):
    """With critical rates (inputs pinned to nominal) the network is critically loaded."""
    for name, path in network_configs.items():
        spec, params, _ = _model(path)
        res = static_planning_lp(
            spec, rates=params.critical_rates,
            pin_inputs=True, nominal_allocation=params.nominal_allocation,
        )
        assert res["success"], name
        assert abs(res["rho_star"] - 1.0) < 1e-6, f"{name}: rho*={res['rho_star']}"


def test_prelimit_underloaded(network_configs):
    """Prelimit rates leave crisscross / PW strictly below capacity (rho < 1)."""
    for name in ("crisscross", "pesic_williams"):
        spec, params, _ = _model(network_configs[name])
        res = static_planning_lp(
            spec, pin_inputs=True, nominal_allocation=params.nominal_allocation
        )
        assert res["success"], name
        assert res["rho_star"] < 1.0 - 1e-6, f"{name}: rho*={res['rho_star']}"
