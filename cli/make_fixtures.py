#!/usr/bin/env python3
"""Generate deterministic fixtures for cross-checking the reflection backends.

Reflects fixed random inputs with the canonical numba LCP backend and stores
(X, Y, H) so tests can assert the numba and GPU backends agree to tight
tolerance. Fixtures are committed under tests/fixtures/, so the test suite does
not depend on regenerating them.

Usage:
    python cli/make_fixtures.py --phase reflection
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401
import numpy as np

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.reflection import get_solver

ROOT = _bootstrap.publication_root()
FIX = os.path.join(ROOT, "tests", "fixtures")

NETWORKS = {
    "crisscross": os.path.join(ROOT, "configs/net_topology/crisscross.yaml"),
    "pesic_williams": os.path.join(ROOT, "configs/net_topology/pesic_williams.yaml"),
    "three_station_bigstep": os.path.join(ROOT, "configs/net_topology/three_station_bigstep.yaml"),
}


def make_reflection_fixtures() -> None:
    os.makedirs(FIX, exist_ok=True)
    for name, path in NETWORKS.items():
        spec, params = load_network(path)
        model = BCPModel(spec, params)
        H = model.H
        rng = np.random.default_rng(20260614)
        X = rng.standard_normal((512, model.I)) * 0.4
        X[:32] = np.abs(X[:32])  # some all-positive passthrough cases
        Y = get_solver("numba", H).project(X)

        out = os.path.join(FIX, f"reflection_{name}.npz")
        np.savez(out, X=X, Y=Y, H=H, source=np.array("numba"))
        # LCP residual of the stored reference
        L = np.linalg.solve(H, (Y - X).T).T
        resid = max(np.maximum(-Y, 0).max(), np.maximum(-L, 0).max(),
                    np.abs(Y * L).max())
        print(f"  {name:24s} -> {out}  (resid={resid:.2e})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="reflection", choices=["reflection"])
    ap.parse_args()
    make_reflection_fixtures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
