#!/usr/bin/env python3
"""Run the prelimit simulator under a baseline or trained BSDE policy.

    python cli/run_prelimit.py --config configs/net_parameters/bsde_bigstep.yaml \
        --policy baseline --smoke
    python cli/run_prelimit.py --config ... --policy bsde --ckpt results/checkpoints/bsde_bigstep

Writes cost decomposition + utilization JSON to results/processed/.
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401

from diffusion_based_policy.config import load_experiment
from diffusion_based_policy.io import write_json
from diffusion_based_policy.policies.baselines import baseline_gradient
from diffusion_based_policy.sim.prelimit import simulate_prelimit

ROOT = _bootstrap.publication_root()


def _gradient_fn(args, exp, model):
    if args.policy == "baseline":
        return baseline_gradient(model, kind=args.baseline_kind), f"baseline:{args.baseline_kind}"
    if args.policy == "bsde":
        from diffusion_based_policy.bsde.evaluator import BSDEEvaluator, load_model_from_checkpoint
        ckpt = args.ckpt or os.path.join(ROOT, "results", "checkpoints", exp.name)
        hidden = tuple(exp.bsde.get("hidden", (100, 100, 100)))
        net = load_model_from_checkpoint(ckpt, model.I, hidden=hidden)
        return BSDEEvaluator(model, net).gradient_fn(), f"bsde:{ckpt}"
    raise ValueError(args.policy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--policy", default="baseline", choices=["baseline", "bsde"])
    ap.add_argument("--baseline-kind", default="linear")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--paths", type=int, default=None)
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--max-jumps", type=int, default=None,
                    help="override prelimit.num_jumps simulation budget")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    exp = load_experiment(args.config)
    model = exp.model()
    pre = exp.prelimit or {}

    grad_fn, policy_desc = _gradient_fn(args, exp, model)

    if args.smoke:
        num_paths = args.paths or 256
        horizon = args.horizon or 0.5
        max_jumps = args.max_jumps  # None -> auto-sized by the simulator
    else:
        num_paths = args.paths or int(pre.get("num_paths", 40000))
        horizon = args.horizon or float(pre.get("horizon", 3.0))
        # prelimit.num_jumps is the simulation budget; honor it as max_jumps.
        max_jumps = args.max_jumps or (int(pre["num_jumps"]) if "num_jumps" in pre else None)
    seed = args.seed if args.seed is not None else exp.seed

    print(f"prelimit: {exp.name} policy={policy_desc} paths={num_paths} "
          f"horizon={horizon} max_jumps={max_jumps} seed={seed}", flush=True)
    res = simulate_prelimit(model, grad_fn, num_paths=num_paths,
                            horizon=horizon, seed=seed, max_jumps=max_jumps)
    res["experiment"] = exp.name
    res["policy"] = policy_desc

    out = os.path.join(ROOT, "results", "processed",
                       f"prelimit_{exp.name}_{args.policy}{'_smoke' if args.smoke else ''}.json")
    write_json(out, res)
    print(f"  cost={res['cost_mean']:.2f} (scaled {res['cost_scaled_mean']:.4f})  "
          f"holding={res['holding']:.2f}  rejection={res['rejection']:.2f}")
    print(f"  utilization={[round(u, 4) for u in res['utilization']]} "
          f"({res['utilization_labels']})")
    print(f"  summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
