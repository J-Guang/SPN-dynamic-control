#!/usr/bin/env python3
"""Compare full reflection backend pool outputs saved by benchmark jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path,
                        default=Path("results/processed/reflection_backend_full"))
    parser.add_argument("--reference", default="exact_enum")
    parser.add_argument("--others", nargs="+", default=["lemke", "pgs"])
    parser.add_argument("--out", type=Path,
                        default=Path("results/processed/reflection_backend_full_compare.json"))
    parser.add_argument("--progress", type=int, default=250)
    return parser.parse_args()


def load_meta(root: Path, backend: str) -> dict:
    with open(root / f"{backend}.json") as f:
        return json.load(f)


def compare_pair(root: Path, ref_meta: dict, other_meta: dict,
                 reference: str, other: str, progress: int) -> dict:
    z_shape = tuple(ref_meta["z_shape"])
    dw_shape = tuple(ref_meta["dw_shape"])
    rz = np.memmap(root / f"{reference}_z.dat", mode="r", dtype=np.float32, shape=z_shape)
    rdw = np.memmap(root / f"{reference}_dw.dat", mode="r", dtype=np.float32, shape=dw_shape)
    oz = np.memmap(root / f"{other}_z.dat", mode="r", dtype=np.float32, shape=z_shape)
    odw = np.memmap(root / f"{other}_dw.dat", mode="r", dtype=np.float32, shape=dw_shape)

    same_z_exact = True
    same_dw_exact = True
    allclose_z = True
    allclose_dw = True
    max_z_diff = 0.0
    max_dw_diff = 0.0
    first_z_diff_segment = None
    first_dw_diff_segment = None
    for i in range(z_shape[0]):
        rz_i = rz[i]
        oz_i = oz[i]
        rdw_i = rdw[i]
        odw_i = odw[i]
        if same_z_exact and not np.array_equal(rz_i, oz_i):
            same_z_exact = False
            first_z_diff_segment = i
        if same_dw_exact and not np.array_equal(rdw_i, odw_i):
            same_dw_exact = False
            first_dw_diff_segment = i
        dz = float(np.max(np.abs(rz_i.astype(np.float64) - oz_i.astype(np.float64))))
        ddw = float(np.max(np.abs(rdw_i.astype(np.float64) - odw_i.astype(np.float64))))
        max_z_diff = max(max_z_diff, dz)
        max_dw_diff = max(max_dw_diff, ddw)
        if allclose_z and not np.allclose(rz_i, oz_i, atol=1e-6, rtol=1e-6):
            allclose_z = False
        if allclose_dw and not np.allclose(rdw_i, odw_i, atol=0.0, rtol=0.0):
            allclose_dw = False
        if (i + 1) % progress == 0 or i + 1 == z_shape[0]:
            print(other, "compare", i + 1, "/", z_shape[0],
                  "max_z_diff", f"{max_z_diff:.3e}",
                  "max_dw_diff", f"{max_dw_diff:.3e}",
                  flush=True)

    return {
        "backend": other,
        "reference": reference,
        "same_z_exact": same_z_exact,
        "same_dw_exact": same_dw_exact,
        "allclose_z_1e-6": allclose_z,
        "allclose_dw_exact": allclose_dw,
        "max_z_diff": max_z_diff,
        "max_dw_diff": max_dw_diff,
        "first_z_diff_segment": first_z_diff_segment,
        "first_dw_diff_segment": first_dw_diff_segment,
        "z_sha256_matches": ref_meta["z_sha256"] == other_meta["z_sha256"],
        "dw_sha256_matches": ref_meta["dw_sha256"] == other_meta["dw_sha256"],
        "speed_ratio_over_reference": (
            other_meta["total_seconds"] / ref_meta["total_seconds"]
        ),
    }


def main() -> None:
    args = parse_args()
    ref_meta = load_meta(args.dir, args.reference)
    comparisons = []
    metas = {args.reference: ref_meta}
    for other in args.others:
        other_meta = load_meta(args.dir, other)
        metas[other] = other_meta
        comparisons.append(compare_pair(
            args.dir, ref_meta, other_meta, args.reference, other, args.progress))

    payload = {
        "reference": args.reference,
        "metadata": metas,
        "comparisons": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(args.out)
    print("COMPARE_RESULT", json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
