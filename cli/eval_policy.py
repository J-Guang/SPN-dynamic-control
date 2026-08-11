#!/usr/bin/env python3
"""Evaluate a trained BSDE policy: Brownian diagnostics + prelimit cost.

    python cli/eval_policy.py --config configs/net_parameters/bsde_crisscross.yaml \
        --ckpt results/checkpoints/bsde_crisscross
    python cli/eval_policy.py --config ... --smoke   # baseline fallback if no ckpt
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401

from diffusion_based_policy.config import load_experiment
from diffusion_based_policy.io import write_json
from diffusion_based_policy.sim.prelimit import simulate_prelimit

ROOT = _bootstrap.publication_root()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--paths", type=int, default=None)
    ap.add_argument("--horizon", type=float, default=None)
    ap.add_argument("--max-jumps", type=int, default=None,
                    help="override prelimit.num_jumps simulation budget")
    ap.add_argument("--b", type=float, default=None, help="override control bound b")
    ap.add_argument("--tag", default=None,
                    help="override checkpoint/output tag (default _b<b>)")
    ap.add_argument("--baseline", default=None,
                    choices=["identity", "linear", "constant", "zero"],
                    help="evaluate a baseline gradient policy instead of a checkpoint "
                         "(identity = MP g=z; linear = MP-h g=h*z)")
    ap.add_argument("--work-conserving", action="store_true",
                    help="processing stations never idle (classical work-conserving "
                         "service); input admission idle/reject is unaffected")
    ap.add_argument("--omega", type=float, default=None,
                    help="override Q-matrix reflection weight omega")
    args = ap.parse_args()

    exp = load_experiment(args.config)
    tag = ""
    if args.b is not None:
        exp.heavy_traffic["b"] = args.b
        tag = f"_b{args.b:g}"
    if args.tag is not None:
        tag = args.tag
    model = exp.model(omega=args.omega)
    pre = exp.prelimit or {}
    hidden = tuple(exp.bsde.get("hidden", (100, 100, 100)))

    diagnostics = {"experiment": exp.name, "b": model.params.b}
    ckpt = args.ckpt or os.path.join(ROOT, "results", "checkpoints", exp.name + tag)

    grad_fn = None
    if args.baseline is not None:
        from diffusion_based_policy.policies.baselines import baseline_gradient
        grad_fn = baseline_gradient(model, args.baseline)
        diagnostics["policy"] = f"baseline:{args.baseline}"
    elif os.path.isdir(ckpt):
        import tensorflow as tf
        if tf.train.latest_checkpoint(ckpt) is not None:
            from diffusion_based_policy.bsde.evaluator import BSDEEvaluator, load_model_from_checkpoint
            net = load_model_from_checkpoint(ckpt, model.I, hidden=hidden)
            ev = BSDEEvaluator(model, net)
            eval_steps = 16 if args.smoke else int(exp.bsde.get("num_steps", 64))
            eval_horizon = 0.5 if args.smoke else float(exp.bsde.get("horizon", 3.0))
            diagnostics["bsde"] = ev.diagnostics(
                num_paths=256 if args.smoke else 2048,
                num_steps=eval_steps,
                horizon=eval_horizon,
                reference_drift=exp.bsde.get("reference_drift", None),
            )
            grad_fn = ev.gradient_fn()
            diagnostics["policy"] = f"bsde:{ckpt}"

    if grad_fn is None:
        from diffusion_based_policy.policies.baselines import baseline_gradient
        grad_fn = baseline_gradient(model)
        diagnostics["policy"] = "baseline:linear (no checkpoint found)"

    num_paths = args.paths or (256 if args.smoke else int(pre.get("num_paths", 40000)))
    horizon = args.horizon or (0.5 if args.smoke else float(pre.get("horizon", 3.0)))
    # honor prelimit.num_jumps as the simulation budget for full runs (CLI override wins)
    if args.max_jumps is not None:
        max_jumps = args.max_jumps
    elif args.smoke:
        max_jumps = None
    else:
        max_jumps = int(pre["num_jumps"]) if "num_jumps" in pre else None
    res = simulate_prelimit(model, grad_fn, num_paths=num_paths,
                            horizon=horizon, seed=exp.seed, max_jumps=max_jumps,
                            work_conserving=args.work_conserving)
    diagnostics["work_conserving"] = bool(args.work_conserving)
    diagnostics["prelimit"] = {
        "cost_mean": res["cost_mean"], "cost_stderr": res.get("cost_stderr", 0.0),
        "cost_scaled_mean": res["cost_scaled_mean"],
        "holding": res["holding"], "rejection": res["rejection"],
        "utilization": res["utilization"], "utilization_labels": res["utilization_labels"],
        "idle_when_servable": res["idle_when_servable"],
        "reject_rate_by_stream": res["reject_rate_by_stream"],
        "reject_stream_labels": res["reject_stream_labels"],
        "reject_fraction": res["reject_fraction"],
    }

    out = os.path.join(ROOT, "results", "processed",
                       f"eval_{exp.name}{tag}{'_smoke' if args.smoke else ''}.json")
    write_json(out, diagnostics)
    print(f"{exp.name}: {diagnostics['policy']}")
    if "bsde" in diagnostics:
        print(f"  v0={diagnostics['bsde']['v0']:.5f}")
    print(f"  prelimit cost={res['cost_mean']:.2f} (scaled {res['cost_scaled_mean']:.4f})")
    print(f"  summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
