"""Pytest configuration: make src/ importable and expose shared fixtures."""
from __future__ import annotations

import os
import sys

# Cap thread pools before numpy / numba / tensorflow load (see diffusion_based_policy._env).
for _k, _v in {
    "NUMBA_NUM_THREADS": "4", "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "TF_NUM_INTEROP_THREADS": "2", "TF_NUM_INTRAOP_THREADS": "4",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}.items():
    os.environ.setdefault(_k, _v)

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)            # publication/
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

NETWORK_CONFIGS = {
    "crisscross": os.path.join(_ROOT, "configs/net_topology/crisscross.yaml"),
    "pesic_williams": os.path.join(_ROOT, "configs/net_topology/pesic_williams.yaml"),
    "three_station_bigstep": os.path.join(_ROOT, "configs/net_topology/three_station_bigstep.yaml"),
}


def publication_root() -> str:
    return _ROOT


@pytest.fixture(scope="session")
def network_configs() -> dict:
    return dict(NETWORK_CONFIGS)


@pytest.fixture(params=list(NETWORK_CONFIGS), scope="session")
def network_name(request) -> str:
    return request.param


@pytest.fixture(scope="session")
def models() -> dict:
    from diffusion_based_policy.bcp import BCPModel
    from diffusion_based_policy.config import load_network

    out = {}
    for name, path in NETWORK_CONFIGS.items():
        spec, params = load_network(path)
        out[name] = BCPModel(spec, params)
    return out
