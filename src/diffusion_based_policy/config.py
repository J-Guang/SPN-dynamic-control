"""Config loading: YAML -> NetworkSpec, BCPParams, BCPModel, and experiments."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml

from .bcp import BCPModel, BCPParams, bcp_params_from_dict, rescale_params
from .network import NetworkSpec, network_from_dict


def _read_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_network(path: str) -> tuple[NetworkSpec, BCPParams]:
    """Load a network YAML into (NetworkSpec, BCPParams)."""
    d = _read_yaml(path)
    spec = network_from_dict(d)
    if "bcp" not in d:
        raise ValueError(f"network config {path} is missing the 'bcp:' block")
    params = bcp_params_from_dict(d["bcp"], spec)
    return spec, params


def load_model(path: str) -> BCPModel:
    spec, params = load_network(path)
    return BCPModel(spec, params)


@dataclass(frozen=True)
class ExperimentConfig:
    """Parsed experiment YAML (Phase 2/3/4 driver settings)."""

    name: str
    network_path: str
    seed: int
    heavy_traffic: dict[str, Any]
    bsde: dict[str, Any]
    reflection: dict[str, Any]
    prelimit: dict[str, Any]
    raw: dict[str, Any]

    def model(self, omega: float | None = None) -> BCPModel:
        spec, params = load_network(self.network_path)
        # allow experiment heavy_traffic to override the scale n / control bound b;
        # mu_hat and gamma are re-derived from them (not config inputs).
        ht = self.heavy_traffic or {}
        n = ht.get("n")
        b = ht.get("b")
        if n is not None or b is not None:
            params = rescale_params(params, spec, n=n, b=b)
        # optional override of the Q-matrix boundary-reflection weight omega
        # (the only free knob in Q; chi stays 0). Used for the omega-sensitivity study.
        if omega is not None:
            import dataclasses
            params = dataclasses.replace(params, omega=float(omega))
        return BCPModel(spec, params)


def load_experiment(path: str) -> ExperimentConfig:
    d = _read_yaml(path)
    base = os.path.dirname(os.path.abspath(path))
    net = d["network"]
    # network path may be relative to the experiment file or to publication/
    if not os.path.isabs(net):
        cand = os.path.join(base, net)
        if not os.path.exists(cand):
            # try relative to the publication/ root (two levels up from experiments/)
            root = os.path.dirname(os.path.dirname(base))
            cand = os.path.join(root, net)
        net = cand

    bsde = d.get("bsde", {}) or {}
    # Fail fast if dt is specified inconsistently with horizon / num_steps. The
    # simulator derives dt = horizon / num_steps, so a stale dt would silently be
    # ignored -- reject it instead of letting "looks changed but isn't" through.
    if "dt" in bsde and "num_steps" in bsde and "horizon" in bsde:
        expected_dt = float(bsde["horizon"]) / int(bsde["num_steps"])
        if abs(float(bsde["dt"]) - expected_dt) > 1e-9 * max(1.0, expected_dt):
            raise ValueError(
                f"{path}: bsde.dt={bsde['dt']} != horizon/num_steps="
                f"{expected_dt} ({bsde['horizon']}/{bsde['num_steps']}); "
                f"dt is derived from horizon and num_steps, fix the config")

    return ExperimentConfig(
        name=str(d.get("name", os.path.splitext(os.path.basename(path))[0])),
        network_path=net,
        seed=int(d.get("seed", 0)),
        heavy_traffic=d.get("heavy_traffic", {}) or {},
        bsde=bsde,
        reflection=d.get("reflection", {}) or {},
        prelimit=d.get("prelimit", {}) or {},
        raw=d,
    )
