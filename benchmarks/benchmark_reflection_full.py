#!/usr/bin/env python3
"""Full-scale reflection backend benchmark for three-station bigstep.

Builds the same continuous reflected-path pool used by the BSDE dataset, then
compares wall-clock time and same-seed numerical equality across reflection
backends.  The exact-enum run is stored as float32 memmaps so later backends can
stream comparisons without keeping multiple 5GB pools in memory.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.reflection.lemke_lcp import LemkeLCPReflection
from diffusion_based_policy.reflection.numba_lcp import NumbaLCPReflection
from diffusion_based_policy.sim.diffusion import _run_segment


def _solver(kind: str, H: np.ndarray):
    if kind == "exact_enum":
        return NumbaLCPReflection(H)
    if kind == "lemke":
        return LemkeLCPReflection(H)
    if kind == "pgs":
        solver = NumbaLCPReflection(H)
        solver._enum_vec = None
        return solver
    raise ValueError(f"unknown backend kind: {kind}")


def _compile(kind: str, model: BCPModel, num_paths: int) -> float:
    solver = _solver(kind, model.H)
    rng = np.random.default_rng(8675309)
    x = -np.abs(rng.standard_normal((num_paths, model.I))) * 0.1
    t0 = time.perf_counter()
    solver.project(x)
    return time.perf_counter() - t0


def _jsonable(result: dict) -> dict:
    out = {}
    for key, value in result.items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def _write_summary(path: Path, results: list[dict]) -> None:
    payload = {
        "results": [_jsonable(r) for r in results],
        "ratios_over_exact_enum": {},
    }
    if results:
        base = results[0]["total_seconds"]
        for result in results[1:]:
            payload["ratios_over_exact_enum"][result["kind"]] = (
                result["total_seconds"] / base
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def build_pool(kind: str, model: BCPModel, args, ref_z=None, ref_dw=None) -> dict:
    dt = args.horizon / args.num_steps
    ref_drift = np.full(model.I, args.reference_drift, dtype=np.float64)
    solver = _solver(kind, model.H)
    rng = np.random.default_rng(args.seed)
    state = np.zeros((args.num_paths, model.I), dtype=np.float64)

    z_shape = (args.pool_segments, args.num_paths, args.num_steps + 1, model.I)
    dw_shape = (args.pool_segments, args.num_paths, args.num_steps, model.I)
    z_path = args.tmp_dir / "ref_z.dat"
    dw_path = args.tmp_dir / "ref_dw.dat"

    z_pool = None
    dw_pool = None
    if kind == "exact_enum":
        z_pool = np.memmap(z_path, mode="w+", dtype=np.float32, shape=z_shape)
        dw_pool = np.memmap(dw_path, mode="w+", dtype=np.float32, shape=dw_shape)

    z_hasher = hashlib.sha256()
    dw_hasher = hashlib.sha256()
    same_z_exact = True
    same_dw_exact = True
    allclose_z = True
    allclose_dw = True
    max_z_diff = 0.0
    max_dw_diff = 0.0
    z_sum = 0.0
    dw_sum = 0.0
    z_max = -np.inf
    dw_max = -np.inf

    print(kind, "warmup_start", flush=True)
    t0 = time.perf_counter()
    warmup_t0 = t0
    for i in range(args.warmup_segments):
        _, _, _, state = _run_segment(
            solver, state, args.num_steps, dt, model.sigma, ref_drift, rng)
        if (i + 1) % args.progress == 0 or i + 1 == args.warmup_segments:
            print(kind, "warmup", i + 1, "/", args.warmup_segments,
                  "seconds", f"{time.perf_counter() - warmup_t0:.1f}",
                  flush=True)

    print(kind, "pool_start", flush=True)
    pool_t0 = time.perf_counter()
    for i in range(args.pool_segments):
        z, dw, _l, state = _run_segment(
            solver, state, args.num_steps, dt, model.sigma, ref_drift, rng)
        z32 = np.ascontiguousarray(z.astype(np.float32))
        dw32 = np.ascontiguousarray(dw.astype(np.float32))

        z_sum += float(z32.sum(dtype=np.float64))
        dw_sum += float(dw32.sum(dtype=np.float64))
        z_max = max(z_max, float(z32.max()))
        dw_max = max(dw_max, float(dw32.max()))
        z_hasher.update(z32.view(np.uint8))
        dw_hasher.update(dw32.view(np.uint8))

        if kind == "exact_enum":
            z_pool[i] = z32
            dw_pool[i] = dw32
        else:
            rz = ref_z[i]
            rdw = ref_dw[i]
            if same_z_exact and not np.array_equal(z32, rz):
                same_z_exact = False
            if same_dw_exact and not np.array_equal(dw32, rdw):
                same_dw_exact = False
            dz = float(np.max(np.abs(z32.astype(np.float64) - rz.astype(np.float64))))
            ddw = float(np.max(np.abs(dw32.astype(np.float64) - rdw.astype(np.float64))))
            max_z_diff = max(max_z_diff, dz)
            max_dw_diff = max(max_dw_diff, ddw)
            if allclose_z and not np.allclose(z32, rz, atol=1e-6, rtol=1e-6):
                allclose_z = False
            if allclose_dw and not np.allclose(dw32, rdw, atol=0.0, rtol=0.0):
                allclose_dw = False

        if (i + 1) % args.progress == 0 or i + 1 == args.pool_segments:
            elapsed = time.perf_counter() - pool_t0
            total = time.perf_counter() - t0
            rate = (i + 1) / elapsed if elapsed > 0.0 else 0.0
            eta = (args.pool_segments - i - 1) / rate if rate > 0.0 else np.nan
            print(kind, "pool", i + 1, "/", args.pool_segments,
                  "pool_seconds", f"{elapsed:.1f}",
                  "total_seconds", f"{total:.1f}",
                  "eta_seconds", f"{eta:.1f}",
                  flush=True)

    if z_pool is not None:
        z_pool.flush()
        dw_pool.flush()
        del z_pool, dw_pool
        gc.collect()

    result = {
        "kind": kind,
        "total_seconds": time.perf_counter() - t0,
        "warmup_segments": args.warmup_segments,
        "pool_segments": args.pool_segments,
        "num_paths": args.num_paths,
        "num_steps": args.num_steps,
        "z_sum": z_sum,
        "dw_sum": dw_sum,
        "z_max": z_max,
        "dw_max": dw_max,
        "z_sha256": z_hasher.hexdigest(),
        "dw_sha256": dw_hasher.hexdigest(),
    }
    if kind != "exact_enum":
        result.update({
            "same_z_exact": same_z_exact,
            "same_dw_exact": same_dw_exact,
            "allclose_z_1e-6": allclose_z,
            "allclose_dw_exact": allclose_dw,
            "max_z_diff": max_z_diff,
            "max_dw_diff": max_dw_diff,
        })
    print("RESULT", json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", default="configs/net_topology/three_station_bigstep.yaml")
    parser.add_argument("--num-paths", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--horizon", type=float, default=0.01)
    parser.add_argument("--warmup-segments", type=int, default=1000)
    parser.add_argument("--pool-segments", type=int, default=5000)
    parser.add_argument("--reference-drift", type=float, default=-0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress", type=int, default=250)
    parser.add_argument("--backends", nargs="+", default=["exact_enum", "lemke", "pgs"])
    parser.add_argument("--tmp-dir", type=Path, default=Path("results/processed/reflection_backend_full_tmp"))
    parser.add_argument("--out", type=Path, default=Path("results/processed/reflection_backend_full.json"))
    parser.add_argument("--keep-reference", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    spec, params = load_network(args.network)
    model = BCPModel(spec, params)
    dataset_gb = (
        (args.pool_segments * args.num_paths * (args.num_steps + 1) * model.I
         + args.pool_segments * args.num_paths * args.num_steps * model.I) * 4
        / 1e9
    )
    print("CONFIG", json.dumps({
        "network": args.network,
        "I": model.I,
        "num_paths": args.num_paths,
        "num_steps": args.num_steps,
        "warmup_segments": args.warmup_segments,
        "pool_segments": args.pool_segments,
        "seed": args.seed,
        "dataset_gb_per_backend": dataset_gb,
        "backends": args.backends,
        "numba_threads": os.environ.get("NUMBA_NUM_THREADS"),
    }, sort_keys=True), flush=True)

    results = []
    compile_times = {}
    for kind in args.backends:
        t = _compile(kind, model, args.num_paths)
        compile_times[kind] = t
        print(kind, "compile_seconds", f"{t:.3f}", flush=True)

    ref_z = None
    ref_dw = None
    try:
        for kind in args.backends:
            if kind == "exact_enum":
                result = build_pool(kind, model, args)
                result["compile_seconds"] = compile_times[kind]
                results.append(result)
                z_shape = (args.pool_segments, args.num_paths,
                           args.num_steps + 1, model.I)
                dw_shape = (args.pool_segments, args.num_paths,
                            args.num_steps, model.I)
                ref_z = np.memmap(args.tmp_dir / "ref_z.dat", mode="r",
                                  dtype=np.float32, shape=z_shape)
                ref_dw = np.memmap(args.tmp_dir / "ref_dw.dat", mode="r",
                                   dtype=np.float32, shape=dw_shape)
            else:
                if ref_z is None or ref_dw is None:
                    raise RuntimeError("exact_enum must run before comparison backends")
                result = build_pool(kind, model, args, ref_z=ref_z, ref_dw=ref_dw)
                result["compile_seconds"] = compile_times[kind]
                results.append(result)
            _write_summary(args.out, results)
        print("SUMMARY_PATH", str(args.out), flush=True)
        print("SUMMARY_JSON", args.out.read_text(), flush=True)
    finally:
        del ref_z, ref_dw
        gc.collect()
        if not args.keep_reference:
            shutil.rmtree(args.tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
