#!/usr/bin/env python3
"""Standalone BCP V_MC evaluation (the diffusion-scale Brownian value), so the
rollout-paths count can be raised without retraining.

Loads a trained checkpoint and re-runs evaluator.brownian_value_mc with the same
settings as the end-of-training call in cli/train_bsde.py (single control
bound b via model.params.b, reference_drift / vmc_dt / vmc_horizon from config),
just with more paths. Writes results/processed/vmc_<exp.name><tag>.json.

    python cli/eval_vmc.py --config CFG --ckpt DIR --b 8 --paths 100000 --tag _b8_100k
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from diffusion_based_policy.config import load_experiment            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--b", type=float, default=None, help="control bound (match training)")
    ap.add_argument("--paths", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--omega", type=float, default=None,
                    help="override Q-matrix reflection weight omega")
    ap.add_argument("--backend", default="numba",
                    help="reflection backend (numba PGS / gpu exact enumeration)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    exp = load_experiment(args.config)
    if args.b is not None:
        exp.heavy_traffic["b"] = args.b
    model = exp.model(omega=args.omega)
    bsde = exp.bsde

    from diffusion_based_policy.bsde.evaluator import BSDEEvaluator, load_model_from_checkpoint
    hidden = tuple(bsde.get("hidden", (100, 100, 100)))
    net = load_model_from_checkpoint(args.ckpt, model.I, hidden=hidden)
    ev = BSDEEvaluator(model, net)

    vmc = ev.brownian_value_mc(
        num_paths=args.paths,
        horizon=float(bsde.get("vmc_horizon", 3.0)),
        dt=float(bsde.get("vmc_dt", 0.01 / 64)),
        reference_drift=bsde.get("reference_drift"),
        control_bound=None,                 # single bound b = model.params.b
        backend=args.backend,
        seed=args.seed,
    )
    out = {
        "experiment": exp.name, "b": float(model.params.b),
        "v0": net.value_at_origin(),
        "v_mc": vmc["v_mc"], "v_mc_stderr": vmc["stderr"],
        "num_paths": vmc["num_paths"], "nsteps": vmc["nsteps"],
        "horizon": vmc["horizon"], "dt": vmc["dt"],
    }
    path = os.path.join(ROOT, "results", "processed", f"vmc_{exp.name}{args.tag}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"{exp.name}: V_NN={out['v0']:.5f}  V_MC={vmc['v_mc']:.5f} +/- {vmc['stderr']:.5f} "
          f"({vmc['num_paths']} paths, {vmc['nsteps']} steps) -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
