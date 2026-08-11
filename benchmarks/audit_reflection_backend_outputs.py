#!/usr/bin/env python3
"""Compute accuracy metrics for saved reflection backend pools.

For each saved pool, reconstructs the unreflected Euler proposal

    X_k = Z_k + m dt + dW_k

and checks the LCP residual of ``Y_k = Z_{k+1}`` against the network reflection
matrix.  Also compares every backend against the exact-enum pool when present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.reflection import lcp_residual


NETWORKS = {
    "TS": ("configs/net_topology/three_station_bigstep.yaml",
           Path("results/processed/reflection_backend_full")),
    "CC": ("configs/net_topology/crisscross.yaml",
           Path("results/processed/reflection_backend_cc")),
    "PW": ("configs/net_topology/pesic_williams.yaml",
           Path("results/processed/reflection_backend_pw")),
}


def load_meta(root: Path, backend: str):
    path = root / f"{backend}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def audit_one(name: str, cfg: str, root: Path, backend: str,
              reference: str = "exact_enum", chunk: int = 25) -> dict | None:
    meta = load_meta(root, backend)
    if meta is None:
        return None

    spec, params = load_network(cfg)
    model = BCPModel(spec, params)
    z_shape = tuple(meta["z_shape"])
    dw_shape = tuple(meta["dw_shape"])
    z = np.memmap(root / f"{backend}_z.dat", mode="r", dtype=np.float32, shape=z_shape)
    dw = np.memmap(root / f"{backend}_dw.dat", mode="r", dtype=np.float32, shape=dw_shape)
    dt = 0.01 / meta["num_steps"]
    drift = np.full(model.I, -0.15, dtype=np.float64)

    max_res = 0.0
    for start in range(0, z_shape[0], chunk):
        stop = min(start + chunk, z_shape[0])
        X = z[start:stop, :, :-1, :].astype(np.float64)
        X += drift * dt
        X += dw[start:stop].astype(np.float64)
        Y = z[start:stop, :, 1:, :].astype(np.float64)
        X2 = X.reshape(-1, model.I)
        Y2 = Y.reshape(-1, model.I)
        res = lcp_residual(X2, Y2, model.H)
        max_res = max(max_res, float(res))

    out = {
        "network": name,
        "backend": backend,
        "total_seconds": meta["total_seconds"],
        "compile_seconds": meta["compile_seconds"],
        "max_lcp_residual": max_res,
        "z_sha256": meta["z_sha256"],
        "dw_sha256": meta["dw_sha256"],
    }

    ref_meta = load_meta(root, reference)
    if ref_meta is not None and backend != reference:
        rz = np.memmap(root / f"{reference}_z.dat", mode="r", dtype=np.float32, shape=z_shape)
        rdw = np.memmap(root / f"{reference}_dw.dat", mode="r", dtype=np.float32, shape=dw_shape)
        max_z_diff = 0.0
        max_dw_diff = 0.0
        same_dw = True
        z_close = True
        for start in range(0, z_shape[0], chunk):
            stop = min(start + chunk, z_shape[0])
            dz = np.max(np.abs(
                z[start:stop].astype(np.float64) - rz[start:stop].astype(np.float64)))
            ddw = np.max(np.abs(
                dw[start:stop].astype(np.float64) - rdw[start:stop].astype(np.float64)))
            max_z_diff = max(max_z_diff, float(dz))
            max_dw_diff = max(max_dw_diff, float(ddw))
            if same_dw and not np.array_equal(dw[start:stop], rdw[start:stop]):
                same_dw = False
            if z_close and not np.allclose(z[start:stop], rz[start:stop], atol=1e-6, rtol=1e-6):
                z_close = False
        out.update({
            "max_z_diff_vs_exact": max_z_diff,
            "max_dw_diff_vs_exact": max_dw_diff,
            "same_dw_as_exact": same_dw,
            "z_allclose_exact_1e-6": z_close,
        })
    elif backend == reference:
        out.update({
            "max_z_diff_vs_exact": 0.0,
            "max_dw_diff_vs_exact": 0.0,
            "same_dw_as_exact": True,
            "z_allclose_exact_1e-6": True,
        })
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+",
                        default=["exact_enum", "lemke", "newton_lemke", "pgs"])
    parser.add_argument("--out", type=Path,
                        default=Path("results/processed/reflection_backend_accuracy_table.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for name, (cfg, root) in NETWORKS.items():
        for backend in args.backends:
            row = audit_one(name, cfg, root, backend)
            if row is not None:
                rows.append(row)
                print("ROW", json.dumps(row, sort_keys=True), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True))
    tmp.replace(args.out)
    print("WROTE", str(args.out), flush=True)


if __name__ == "__main__":
    main()
