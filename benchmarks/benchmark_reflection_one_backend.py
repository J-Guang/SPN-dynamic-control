#!/usr/bin/env python3
"""Build one full-scale reflected-path pool for one reflection backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.reflection.lemke_lcp import LemkeLCPReflection
from diffusion_based_policy.reflection.newton_lemke import NewtonLemkeReflection
from diffusion_based_policy.reflection.numba_lcp import NumbaLCPReflection
from diffusion_based_policy.sim.diffusion import _run_segment


def make_solver(kind: str, H: np.ndarray):
    if kind == "exact_enum":
        return NumbaLCPReflection(H)
    if kind == "lemke":
        return LemkeLCPReflection(H)
    if kind == "newton_lemke":
        return NewtonLemkeReflection(H)
    if kind == "pgs":
        solver = NumbaLCPReflection(H)
        solver._enum_vec = None
        return solver
    raise ValueError(f"unknown backend: {kind}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True,
                        choices=["exact_enum", "lemke", "newton_lemke", "pgs"])
    parser.add_argument("--network", default="configs/net_topology/three_station_bigstep.yaml")
    parser.add_argument("--num-paths", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--horizon", type=float, default=0.01)
    parser.add_argument("--warmup-segments", type=int, default=1000)
    parser.add_argument("--pool-segments", type=int, default=5000)
    parser.add_argument("--reference-drift", type=float, default=-0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", type=int, default=250)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/processed/reflection_backend_full"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spec, params = load_network(args.network)
    model = BCPModel(spec, params)
    dt = args.horizon / args.num_steps
    drift = np.full(model.I, args.reference_drift, dtype=np.float64)

    z_shape = (args.pool_segments, args.num_paths, args.num_steps + 1, model.I)
    dw_shape = (args.pool_segments, args.num_paths, args.num_steps, model.I)
    z_path = args.out_dir / f"{args.backend}_z.dat"
    dw_path = args.out_dir / f"{args.backend}_dw.dat"
    meta_path = args.out_dir / f"{args.backend}.json"

    dataset_gb = ((np.prod(z_shape) + np.prod(dw_shape)) * 4) / 1e9
    print("CONFIG", json.dumps({
        "backend": args.backend,
        "network": args.network,
        "I": model.I,
        "num_paths": args.num_paths,
        "num_steps": args.num_steps,
        "warmup_segments": args.warmup_segments,
        "pool_segments": args.pool_segments,
        "seed": args.seed,
        "dataset_gb": dataset_gb,
        "numba_threads": os.environ.get("NUMBA_NUM_THREADS"),
        "z_path": str(z_path),
        "dw_path": str(dw_path),
    }, sort_keys=True), flush=True)

    solver = make_solver(args.backend, model.H)
    compile_x = -np.abs(np.random.default_rng(123).standard_normal((args.num_paths, model.I))) * 0.1
    compile_t0 = time.perf_counter()
    solver.project(compile_x)
    compile_seconds = time.perf_counter() - compile_t0
    print(args.backend, "compile_seconds", f"{compile_seconds:.3f}", flush=True)

    rng = np.random.default_rng(args.seed)
    state = np.zeros((args.num_paths, model.I), dtype=np.float64)
    z_pool = np.memmap(z_path, mode="w+", dtype=np.float32, shape=z_shape)
    dw_pool = np.memmap(dw_path, mode="w+", dtype=np.float32, shape=dw_shape)

    z_hash = hashlib.sha256()
    dw_hash = hashlib.sha256()
    z_sum = 0.0
    dw_sum = 0.0
    z_max = -np.inf
    dw_max = -np.inf

    total_t0 = time.perf_counter()
    warmup_t0 = total_t0
    print(args.backend, "warmup_start", flush=True)
    for i in range(args.warmup_segments):
        _, _, _, state = _run_segment(
            solver, state, args.num_steps, dt, model.sigma, drift, rng)
        if (i + 1) % args.progress == 0 or i + 1 == args.warmup_segments:
            print(args.backend, "warmup", i + 1, "/", args.warmup_segments,
                  "seconds", f"{time.perf_counter() - warmup_t0:.1f}",
                  flush=True)

    pool_t0 = time.perf_counter()
    print(args.backend, "pool_start", flush=True)
    for i in range(args.pool_segments):
        z, dw, _l, state = _run_segment(
            solver, state, args.num_steps, dt, model.sigma, drift, rng)
        z32 = np.ascontiguousarray(z.astype(np.float32))
        dw32 = np.ascontiguousarray(dw.astype(np.float32))
        z_pool[i] = z32
        dw_pool[i] = dw32

        z_hash.update(z32.view(np.uint8))
        dw_hash.update(dw32.view(np.uint8))
        z_sum += float(z32.sum(dtype=np.float64))
        dw_sum += float(dw32.sum(dtype=np.float64))
        z_max = max(z_max, float(z32.max()))
        dw_max = max(dw_max, float(dw32.max()))

        if (i + 1) % args.progress == 0 or i + 1 == args.pool_segments:
            elapsed = time.perf_counter() - pool_t0
            total = time.perf_counter() - total_t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (args.pool_segments - i - 1) / rate if rate > 0 else np.nan
            print(args.backend, "pool", i + 1, "/", args.pool_segments,
                  "pool_seconds", f"{elapsed:.1f}",
                  "total_seconds", f"{total:.1f}",
                  "eta_seconds", f"{eta:.1f}",
                  flush=True)

    z_pool.flush()
    dw_pool.flush()
    total_seconds = time.perf_counter() - total_t0
    result = {
        "backend": args.backend,
        "compile_seconds": compile_seconds,
        "total_seconds": total_seconds,
        "network": args.network,
        "I": model.I,
        "num_paths": args.num_paths,
        "num_steps": args.num_steps,
        "warmup_segments": args.warmup_segments,
        "pool_segments": args.pool_segments,
        "seed": args.seed,
        "z_shape": z_shape,
        "dw_shape": dw_shape,
        "z_path": str(z_path),
        "dw_path": str(dw_path),
        "z_sha256": z_hash.hexdigest(),
        "dw_sha256": dw_hash.hexdigest(),
        "z_sum": z_sum,
        "dw_sum": dw_sum,
        "z_max": z_max,
        "dw_max": dw_max,
    }
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    tmp.replace(meta_path)
    print("RESULT", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
