#!/usr/bin/env python3
"""Train a BSDE value/gradient network from an experiment config.

    python cli/train_bsde.py --config configs/net_parameters/bsde_crisscross.yaml
    python cli/train_bsde.py --config ... --smoke      # tiny run for CI

Checkpoints go to results/checkpoints/<name>/ and metrics to
results/processed/<name>/.
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401

from diffusion_based_policy.config import load_experiment
from diffusion_based_policy.io import write_json

ROOT = _bootstrap.publication_root()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--backend", default=None, help="override reflection backend")
    ap.add_argument("--b", type=float, default=None, help="override control bound b")
    ap.add_argument("--omega", type=float, default=None,
                    help="override Q-matrix reflection weight omega (sensitivity study)")
    ap.add_argument("--tag", default=None,
                    help="override output/checkpoint tag (default _b<b>)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start: restore net weights from this checkpoint dir")
    ap.add_argument("--fixed-b", action="store_true",
                    help="disable the b-curriculum ramp; train at constant b")
    ap.add_argument("--curriculum-init", type=float, default=None,
                    help="override bound_curriculum initial_bound (the b_eff ramp start)")
    ap.add_argument("--lr", type=float, default=None,
                    help="override LR peak (per-stage warm-restart)")
    ap.add_argument("--behavior-from", default=None,
                    help="on-policy reference: generate the training pool under the "
                         "frozen policy in this checkpoint dir (drift zeta + theta R^T)")
    ap.add_argument("--behavior-bound", type=float, default=None,
                    help="control bound 'a' of the --behavior-from policy (default: --b)")
    ap.add_argument("--on-policy", action="store_true",
                    help="TRUE on-policy: reference pool driven by the LIVE net, "
                         "refreshed periodically (50%% on-policy + 50%% offline anchor)")
    ap.add_argument("--on-policy-refresh", type=int, default=5000,
                    help="steps between on-policy pool refreshes")
    ap.add_argument("--on-policy-ratio", type=float, default=0.5,
                    help="fraction of on-policy (vs offline-anchor) paths")
    ap.add_argument("--on-policy-pool", type=int, default=0,
                    help="on-policy pool segments (0 -> train_steps_per_epoch)")
    ap.add_argument("--b-sched", type=float, default=None,
                    help="split control bound: b for scheduling (processing) blocks")
    ap.add_argument("--b-reject", type=float, default=None,
                    help="split control bound: b for rejection/routing (input) blocks")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the experiment seed (for multi-seed sweeps)")
    args = ap.parse_args()

    exp = load_experiment(args.config)
    eff_seed = args.seed if args.seed is not None else exp.seed   # multi-seed sweeps
    bsde = exp.bsde or {}
    backend = args.backend or exp.reflection.get("backend", "numba")
    # control-bound sweep: override b (and tag outputs with it)
    tag = ""
    if args.b is not None:
        exp.heavy_traffic["b"] = args.b
        tag = f"_b{args.b:g}"
    if args.tag is not None:
        tag = args.tag

    from diffusion_based_policy.bsde.model import ValueGradModel
    from diffusion_based_policy.bsde.schedule import (
        bound_curriculum_kwargs,
        lr_schedule_kwargs,
        resolve_training_updates,
    )
    from diffusion_based_policy.bsde.trainer import BSDETrainer, TrainConfig

    model = exp.model(omega=args.omega)

    # staged warm-restart knobs: --fixed-b disables the in-run b ramp (each stage
    # trains at constant b, warm-started from the previous stage), --lr overrides
    # the cosine LR peak so a fresh optimizer at each b restarts from a usable LR.
    curr_kwargs = ({"bound_curriculum": "none"} if args.fixed_b
                   else bound_curriculum_kwargs(bsde))
    if args.curriculum_init is not None and not args.fixed_b:
        curr_kwargs["curriculum_initial_bound"] = float(args.curriculum_init)
    lr_peak = args.lr if args.lr is not None else None

    if args.smoke:
        hidden = tuple(bsde.get("hidden_smoke", (32, 32)))
        cfg = TrainConfig(
            num_paths=int(bsde.get("smoke_paths", 64)),
            num_steps=int(bsde.get("smoke_steps", 16)),
            horizon=float(bsde.get("smoke_horizon", 0.5)),
            steps=args.steps or 50, lr=float(bsde.get("lr", 1e-3)),
            **lr_schedule_kwargs(bsde),
            **bound_curriculum_kwargs(bsde),
            backend=backend, hidden=hidden, seed=eff_seed,
            val_interval=25, ckpt_interval=0,
            reference_drift=bsde.get("reference_drift", None),
            loss_scale_by_horizon_sq=bool(bsde.get("loss_scale_by_horizon_sq", True)),
            sampling_mode=bsde.get("sampling_mode", "continuous"),
            warmup_segments=0,   # no burn-in for the tiny smoke run
            log_dir=os.path.join(ROOT, "results", "processed", exp.name + "_smoke"),
        )
    else:
        ckpt_dir = os.path.join(ROOT, "results", "checkpoints", exp.name + tag)
        log_dir = os.path.join(ROOT, "results", "processed", exp.name + tag)
        cfg = TrainConfig(
            num_paths=int(bsde.get("num_paths", 1024)),
            num_steps=int(bsde.get("num_steps", 64)),
            horizon=float(bsde.get("horizon", 3.0)),
            steps=resolve_training_updates(bsde, args.steps),
            lr=lr_peak if lr_peak is not None else float(bsde.get("lr", 5e-4)),
            **lr_schedule_kwargs(bsde),
            **curr_kwargs,
            weight_decay=float(bsde.get("weight_decay", 1e-3)),
            clipnorm=float(bsde.get("clipnorm", 50.0)),
            grad_nonneg_weight=float(bsde.get("grad_nonneg_weight", 0.0)),
            backend=backend,
            hidden=tuple(bsde.get("hidden", (100, 100, 100))),
            seed=eff_seed,
            loss_scale_by_horizon_sq=bool(bsde.get("loss_scale_by_horizon_sq", True)),
            val_interval=int(bsde.get("val_interval", 200)),
            val_paths=int(bsde.get("val_paths", 2048)),
            log_interval=int(bsde.get("log_interval", 20)),
            ckpt_interval=int(bsde.get("ckpt_interval", 1000)),
            init=bsde.get("init", "zeros"),
            init_scale=float(bsde.get("init_scale", 1.0)),
            reference_drift=bsde.get("reference_drift", None),
            sampling_mode=bsde.get("sampling_mode", "continuous"),
            warmup_segments=int(bsde.get("warmup_segments", 200)),
            # training-segment pool reused across epochs = train_steps_per_epoch
            # (the published runs warm up 1000 segments, then train on
            # num_iterations - 1000; our train_steps_per_epoch already equals
            # that training count, and validation is a separate fixed batch).
            pool_segments=int(bsde.get(
                "pool_segments", int(bsde.get("train_steps_per_epoch", 5000)))),
            ckpt_dir=ckpt_dir, log_dir=log_dir,
        )

    net = ValueGradModel(model.I, hidden=cfg.hidden, seed=cfg.seed)
    if args.init_from:
        import numpy as _np
        import tensorflow as _tf
        z0 = _np.zeros((1, model.I), _np.float32)
        net.value(z0); net.grad(z0)            # build variables before restore
        latest = _tf.train.latest_checkpoint(args.init_from)
        if latest is None:
            raise SystemExit(f"--init-from: no checkpoint found in {args.init_from}")
        _tf.train.Checkpoint(model=net).restore(latest).expect_partial()
        print(f"warm-start: restored net weights from {latest}", flush=True)

    if args.behavior_from and not args.smoke:
        import numpy as _np
        import tensorflow as _tf
        bnet = ValueGradModel(model.I, hidden=cfg.hidden, seed=cfg.seed + 13)
        z0 = _np.zeros((1, model.I), _np.float32)
        bnet.value(z0); bnet.grad(z0)            # build variables before restore
        blatest = _tf.train.latest_checkpoint(args.behavior_from)
        if blatest is None:
            raise SystemExit(f"--behavior-from: no checkpoint found in {args.behavior_from}")
        _tf.train.Checkpoint(model=bnet).restore(blatest).expect_partial()

        @_tf.function(reduce_retracing=True)
        def _bgrad(z):
            return bnet.grad(z)

        def behavior_grad_fn(z_np):
            return _bgrad(_tf.constant(z_np, _tf.float32)).numpy()

        cfg.behavior_grad_fn = behavior_grad_fn
        cfg.behavior_bound = (args.behavior_bound if args.behavior_bound is not None
                              else float(model.params.b))
        print(f"on-policy reference: behaviour net from {blatest}, "
              f"a={cfg.behavior_bound:g}", flush=True)

    if args.on_policy and not args.smoke:
        cfg.on_policy = True
        cfg.on_policy_ratio = float(args.on_policy_ratio)
        cfg.on_policy_refresh_every = int(args.on_policy_refresh)
        cfg.on_policy_pool = int(args.on_policy_pool)
        print(f"TRUE on-policy: live net reference, refresh every "
              f"{cfg.on_policy_refresh_every} steps, ratio {cfg.on_policy_ratio}, "
              f"pool {cfg.on_policy_pool or 'default'}", flush=True)

    if args.b_sched is not None and args.b_reject is not None and not args.smoke:
        cfg.b_sched = float(args.b_sched)
        cfg.b_reject = float(args.b_reject)
        print(f"split control bound: b_sched={cfg.b_sched:g} (scheduling) / "
              f"b_reject={cfg.b_reject:g} (rejection+routing)", flush=True)
    epoch_info = ""
    if not args.smoke and "train_steps_per_epoch" in bsde and "train_epochs" in bsde:
        epoch_info = f" ({bsde['train_steps_per_epoch']} x {bsde['train_epochs']} epochs)"
    print(f"training {exp.name}: I={model.I} updates={cfg.steps}{epoch_info} "
          f"paths={cfg.num_paths} backend={backend}", flush=True)
    trainer = BSDETrainer(model, net, cfg)
    history = trainer.train()

    # BCP Brownian value by simulation rollout (the paper's "BCP V_MC"), run by
    # default after training. Smoke uses a cheap short rollout.
    from diffusion_based_policy.bsde.evaluator import BSDEEvaluator
    ev = BSDEEvaluator(model, net)
    if args.smoke:
        vmc = ev.brownian_value_mc(num_paths=64, horizon=0.1, dt=0.01 / 16,
                                   reference_drift=bsde.get("reference_drift"),
                                   backend=backend, seed=eff_seed + 777)
    else:
        vmc = ev.brownian_value_mc(
            num_paths=int(bsde.get("vmc_paths", 2048)),
            horizon=float(bsde.get("vmc_horizon", 3.0)),
            dt=float(bsde.get("vmc_dt", 0.01 / 64)),
            reference_drift=bsde.get("reference_drift"),
            control_bound=((cfg.b_sched, cfg.b_reject)
                           if cfg.b_sched is not None and cfg.b_reject is not None
                           else None),
            backend=backend, seed=eff_seed + 777)
    print(f"BCP V_MC(0) = {vmc['v_mc']:.5f} +/- {vmc['stderr']:.5f} "
          f"(rollout horizon={vmc['horizon']}, {vmc['nsteps']} steps, "
          f"{vmc['num_paths']} paths)", flush=True)

    summary = {
        "experiment": exp.name,
        "smoke": args.smoke,
        "v0": net.value_at_origin(),
        "v_mc": vmc["v_mc"],
        "v_mc_stderr": vmc["stderr"],
        "final_loss": history[-1]["loss"] if history else None,
        "final_residual": history[-1]["residual_sq"] if history else None,
        "steps": cfg.steps,
        "train_steps_per_epoch": bsde.get("train_steps_per_epoch"),
        "train_epochs": bsde.get("train_epochs"),
        "checkpoint_dir": cfg.ckpt_dir or None,
    }
    summary["b"] = model.params.b
    out = os.path.join(ROOT, "results", "processed",
                       f"train_{exp.name}{tag}{'_smoke' if args.smoke else ''}.json")
    write_json(out, summary)
    print(f"\nb={model.params.b:g}  v0={summary['v0']:.5f}  "
          f"v_mc={summary['v_mc']:.5f}  final_loss={summary['final_loss']}")
    print(f"summary -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
