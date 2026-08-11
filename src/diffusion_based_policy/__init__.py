"""diffusion_based_policy: publication implementation of the diffusion-based BCP policy.

Public surface for Phase 0 (foundation). Heavier submodules (reflection, bsde,
policies, sim) are imported lazily by their own scripts to keep optional
dependencies (numba, tensorflow) out of the import path until needed.
"""
from __future__ import annotations

from . import _env  # noqa: F401  (set thread caps before numpy/numba/tf load)
from .bcp import BCPModel, BCPParams, bcp_params_from_dict
from .config import (
    ExperimentConfig,
    load_experiment,
    load_model,
    load_network,
)
from .network import NetworkSpec, network_from_dict

__all__ = [
    "NetworkSpec",
    "network_from_dict",
    "BCPModel",
    "BCPParams",
    "bcp_params_from_dict",
    "load_network",
    "load_model",
    "load_experiment",
    "ExperimentConfig",
]

__version__ = "0.1.0"
