"""Custom TensorFlow BSDE training loop (no model.fit / model.compile).

Uses tf.GradientTape, a pure-TF AdamW optimizer, tf.train.Checkpoint, and
CSV/JSON metric logging. Sample generation, model, loss, and checkpointing are
kept in separate modules; this file only orchestrates the loop.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import tensorflow as tf

from ..bcp import BCPModel
from ..io import append_csv_row, ensure_dir, write_json
from ..metrics import MetricLog
from .dataset import (
    ContinuousBatchSampler,
    LiveHybridPool,
    PooledContinuousSampler,
    PooledOnPolicySampler,
    make_training_batch,
    make_validation_batch,
)
from .losses import HamiltonianTF, bsde_loss
from .model import ValueGradModel
from .optim import AdamW
from .schedule import bound_curriculum_kwargs, effective_control_bound
from .schedule import learning_rate_at as scheduled_learning_rate
from .schedule import lr_schedule_kwargs


@dataclass
class TrainConfig:
    num_paths: int = 1024
    num_steps: int = 64
    horizon: float = 3.0
    steps: int = 2000
    lr: float = 5e-4
    lr_schedule: str = "cosine"
    lr_min_ratio: float = 1e-2
    lr_warmup_steps: int = 0
    lr_decay_steps: int | None = None
    lr_piecewise_boundaries: tuple[int, ...] = field(default_factory=tuple)
    lr_piecewise_values: tuple[float, ...] = field(default_factory=tuple)
    weight_decay: float = 1e-3
    clipnorm: float = 50.0
    backend: str = "numba"
    init: str = "zeros"
    init_scale: float = 1.0
    sampling_mode: str = "continuous"   # "continuous" (chained path) or "iid"
    warmup_segments: int = 0            # burn-in segments before the first batch
    pool_segments: int = 0             # >0: pre-generate this many segments and
    #                                    reuse them (shuffled per epoch), like the
    #                                    published runs; 0: fresh segment per step
    reference_drift: list | None = None
    loss_scale_by_horizon_sq: bool = True
    bound_curriculum: str = "none"
    curriculum_pace: float = 0.0
    curriculum_initial_bound: float = 1.0
    grad_nonneg_weight: float = 0.0
    hidden: tuple = (100, 100, 100)
    seed: int = 0
    val_interval: int = 100
    val_paths: int = 1024
    log_interval: int = 20
    ckpt_interval: int = 0           # 0 disables checkpointing
    ckpt_dir: str = ""
    log_dir: str = ""
    lr_decay: float = 1.0            # multiplicative decay applied at each val step
    # on-policy reference sampling (lagged/fitted): when set, the training pool is
    # generated under this frozen behaviour policy (drift mu = zeta + theta R^T)
    # instead of the constant offline reference drift.
    behavior_grad_fn: object = None   # callable z(P,I)->g(P,I) (numpy)
    behavior_bound: float = 1.0       # control bound 'a' of the behaviour policy
    # TRUE (live) on-policy: pool generated under the LIVE net (refreshed every
    # ``on_policy_refresh_every`` steps), 50/50 on-policy + offline anchor.
    on_policy: bool = False
    on_policy_ratio: float = 0.5
    on_policy_refresh_every: int = 5000
    on_policy_pool: int = 0           # 0 -> use pool_segments
    on_policy_warmup_segments: int = 200
    # split control bound: different b for scheduling (processing blocks) vs
    # rejection/routing (input blocks). When both set, overrides the curriculum
    # with a constant (b_sched, b_reject) pair.
    b_sched: float | None = None
    b_reject: float | None = None


class BSDETrainer:
    def __init__(self, model: BCPModel, net: ValueGradModel, cfg: TrainConfig):
        self.model = model
        self.net = net
        self.cfg = cfg
        self.ham = HamiltonianTF(model, dtype=net.dtype_)
        self.opt = AdamW(net.trainable_variables, lr=cfg.lr,
                         weight_decay=cfg.weight_decay, clipnorm=cfg.clipnorm)
        self._ref_drift = (np.asarray(cfg.reference_drift, float)
                           if cfg.reference_drift is not None else None)
        self._base_seed = cfg.seed + 1
        # Continuous (chained) sampling by default: one long reflected path per
        # row, warmed up to the stationary distribution, so segments start from
        # the whole occupied region rather than always from the origin.
        self._cont_sampler = None
        self._live_pool = None
        self._live_grad_fn = None
        if cfg.on_policy:
            # TRUE on-policy: pool driven by the LIVE net, refreshed periodically.
            self._live_grad_fn = self._make_live_grad_fn()
            self._live_pool = LiveHybridPool(
                model, num_paths=cfg.num_paths, num_steps=cfg.num_steps,
                horizon=cfg.horizon,
                pool_segments=cfg.on_policy_pool if cfg.on_policy_pool > 0
                else (cfg.pool_segments if cfg.pool_segments > 0 else 5000),
                behavior_bound=float(model.params.b), behavior_ratio=cfg.on_policy_ratio,
                reference_drift=self._ref_drift, backend=cfg.backend,
                init=cfg.init, init_scale=cfg.init_scale, seed=self._base_seed,
                warmup_segments=cfg.on_policy_warmup_segments)
            self._cont_sampler = self._live_pool
        elif (cfg.sampling_mode or "continuous").lower() == "continuous":
            if cfg.behavior_grad_fn is not None:
                # lagged on-policy: fixed pool under a frozen behaviour policy
                self._cont_sampler = PooledOnPolicySampler(
                    model, num_paths=cfg.num_paths, num_steps=cfg.num_steps,
                    horizon=cfg.horizon,
                    pool_segments=cfg.pool_segments if cfg.pool_segments > 0 else 5000,
                    grad_fn=cfg.behavior_grad_fn, behavior_bound=cfg.behavior_bound,
                    backend=cfg.backend, init=cfg.init, init_scale=cfg.init_scale,
                    seed=self._base_seed, warmup_segments=cfg.warmup_segments)
            elif cfg.pool_segments and cfg.pool_segments > 0:
                # bounded pool reused across epochs (fast, published scheme)
                self._cont_sampler = PooledContinuousSampler(
                    model, num_paths=cfg.num_paths, num_steps=cfg.num_steps,
                    horizon=cfg.horizon, pool_segments=cfg.pool_segments,
                    backend=cfg.backend, reference_drift=self._ref_drift,
                    init=cfg.init, init_scale=cfg.init_scale, seed=self._base_seed,
                    warmup_segments=cfg.warmup_segments)
            else:
                # fresh segment per step (simple; use for small/smoke runs)
                self._cont_sampler = ContinuousBatchSampler(
                    model, num_paths=cfg.num_paths, num_steps=cfg.num_steps,
                    horizon=cfg.horizon, backend=cfg.backend,
                    reference_drift=self._ref_drift, init=cfg.init,
                    init_scale=cfg.init_scale, seed=self._base_seed,
                    warmup_segments=cfg.warmup_segments)
        self._val_batch = make_validation_batch(
            model, num_paths=cfg.val_paths, num_steps=cfg.num_steps,
            horizon=cfg.horizon, backend=cfg.backend, reference_drift=self._ref_drift,
            init=cfg.init, init_scale=cfg.init_scale,
        )
        self.metrics = MetricLog()
        self.ckpt = None
        self.ckpt_manager = None
        if cfg.ckpt_interval > 0 and cfg.ckpt_dir:
            ensure_dir(cfg.ckpt_dir)
            self.ckpt = tf.train.Checkpoint(model=net, optimizer=self.opt)
            self.ckpt_manager = tf.train.CheckpointManager(
                self.ckpt, cfg.ckpt_dir, max_to_keep=3)

    def _make_live_grad_fn(self):
        """numpy callable z(P,I) -> g(P,I) wrapping the LIVE net (current weights)."""
        net = self.net
        dt = net.dtype_

        @tf.function(reduce_retracing=True)
        def _g(z):
            return net.grad(z)

        def grad_fn(z_np):
            return _g(tf.constant(z_np, dt)).numpy()
        return grad_fn

    # ----------------------------------------------------------------- lr
    def learning_rate_at(self, step: int) -> float:
        return scheduled_learning_rate(
            step,
            base_lr=self.cfg.lr,
            steps=self.cfg.steps,
            schedule=self.cfg.lr_schedule,
            min_ratio=self.cfg.lr_min_ratio,
            warmup_steps=self.cfg.lr_warmup_steps,
            decay_steps=self.cfg.lr_decay_steps,
            piecewise_boundaries=self.cfg.lr_piecewise_boundaries,
            piecewise_values=self.cfg.lr_piecewise_values,
        )

    def control_bound_at(self, step: int):
        """Effective control bound at this step: a scalar b_eff, or a
        (b_sched, b_reject) pair when split bounds are configured."""
        kw = dict(
            dim=self.model.I,
            curriculum=self.cfg.bound_curriculum,
            pace=self.cfg.curriculum_pace,
            initial_bound=self.cfg.curriculum_initial_bound,
        )
        if self.cfg.b_sched is not None and self.cfg.b_reject is not None:
            # split bound: ramp each component to its target via the curriculum
            # (curriculum="none" -> constant = the fixed-b split case).
            be_s = effective_control_bound(step, target_bound=float(self.cfg.b_sched), **kw)
            be_r = effective_control_bound(step, target_bound=float(self.cfg.b_reject), **kw)
            return (be_s, be_r)
        return effective_control_bound(step, target_bound=self.model.params.b, **kw)

    # ----------------------------------------------------------------- step
    def _train_step(self, batch, control_bound: float) -> dict:
        with tf.GradientTape() as tape:
            loss, comps = bsde_loss(self.net, batch, self.model,
                                    hamiltonian=self.ham,
                                    grad_nonneg_weight=self.cfg.grad_nonneg_weight,
                                    control_bound=control_bound,
                                    scale_by_horizon_sq=self.cfg.loss_scale_by_horizon_sq)
        grads = tape.gradient(loss, self.net.trainable_variables)
        self.opt.apply_gradients(list(zip(grads, self.net.trainable_variables)))
        return {k: float(v.numpy()) for k, v in comps.items()}

    def validate(self, control_bound: float) -> dict:
        loss, comps = bsde_loss(self.net, self._val_batch, self.model,
                                hamiltonian=self.ham,
                                control_bound=control_bound,
                                scale_by_horizon_sq=self.cfg.loss_scale_by_horizon_sq)
        return {f"val_{k}": float(v.numpy()) for k, v in comps.items()}

    # ----------------------------------------------------------------- loop
    def train(self) -> list[dict]:
        cfg = self.cfg
        history: list[dict] = []
        csv_path = os.path.join(cfg.log_dir, "metrics.csv") if cfg.log_dir else None
        if cfg.log_dir:
            ensure_dir(cfg.log_dir)
        fields = ["step", "loss", "residual_sq", "residual_sq_raw", "v0_mean",
                  "val_residual_sq", "lr", "control_bound", "sec"]
        t0 = time.time()
        for step in range(cfg.steps):
            self.opt.set_lr(self.learning_rate_at(step))
            control_bound = self.control_bound_at(step)
            if self._live_pool is not None and step % cfg.on_policy_refresh_every == 0:
                # regenerate the reference pool under the CURRENT live policy
                self._live_pool.refresh(self._live_grad_fn, behavior_bound=control_bound)
                if cfg.log_interval and step % cfg.log_interval == 0:
                    print(f"  [on-policy] refreshed pool #{self._live_pool.refreshes} "
                          f"at step {step} (a={control_bound:.2f})", flush=True)
            if self._cont_sampler is not None:
                batch = self._cont_sampler.batch()           # continuous chained path
            else:
                batch = make_training_batch(                 # iid restart-from-init
                    self.model, num_paths=cfg.num_paths, num_steps=cfg.num_steps,
                    horizon=cfg.horizon, seed=self._base_seed + step,
                    backend=cfg.backend, reference_drift=self._ref_drift,
                    init=cfg.init, init_scale=cfg.init_scale,
                )
            comps = self._train_step(batch, control_bound=control_bound)
            row = {"step": step, "loss": comps["total"],
                   "residual_sq": comps["residual_sq"],
                   "residual_sq_raw": comps["residual_sq_raw"],
                   "v0_mean": comps["v0_mean"],
                   "lr": float(self.opt.lr.numpy()),
                   # store a scalar for the CSV/metrics; the loss gets the full
                   # (possibly split) control_bound directly above.
                   "control_bound": (float(control_bound[0])
                                     if isinstance(control_bound, (tuple, list))
                                     else control_bound)}

            if cfg.val_interval and step % cfg.val_interval == 0:
                row.update(self.validate(control_bound=control_bound))
                if cfg.lr_decay != 1.0 and (cfg.lr_schedule or "").lower() in {"constant", "none", "fixed"}:
                    self.opt.set_lr(float(self.opt.lr.numpy()) * cfg.lr_decay)

            if cfg.ckpt_interval and self.ckpt_manager and step % cfg.ckpt_interval == 0:
                self.ckpt_manager.save()

            if step % cfg.log_interval == 0 or step == cfg.steps - 1:
                row["sec"] = round(time.time() - t0, 2)
                self.metrics.log(step, **{k: v for k, v in row.items() if k != "step"})
                history.append(row)
                if csv_path:
                    append_csv_row(csv_path, row, fields)
                cb = row['control_bound']
                cb_str = (f"{cb[0]:.1f}/{cb[1]:.1f}" if isinstance(cb, (tuple, list))
                          else f"{cb:.3f}")
                msg = (f"  step {step:6d}  loss={row['loss']:.5e}  "
                       f"resid={row['residual_sq']:.5e}  "
                       f"v0={row['v0_mean']:.5f}  lr={row['lr']:.3e}  "
                       f"b_eff={cb_str}")
                if "val_residual_sq" in row:
                    msg += f"  val_resid={row['val_residual_sq']:.5e}"
                print(msg, flush=True)

        if self.ckpt_manager:
            self.ckpt_manager.save()
        result = {
            "final_loss": history[-1]["loss"] if history else float("nan"),
            "final_residual": history[-1]["residual_sq"] if history else float("nan"),
            "v0": self.net.value_at_origin(),
            "steps": cfg.steps,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        if cfg.log_dir:
            write_json(os.path.join(cfg.log_dir, "summary.json"),
                       {"config": asdict_safe(cfg), "result": result,
                        "history": history})
        return history


def asdict_safe(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    d["hidden"] = list(cfg.hidden)
    # drop runtime callables (not JSON serializable); record presence instead
    d.pop("behavior_grad_fn", None)
    d["on_policy_reference"] = cfg.behavior_grad_fn is not None
    return d


def smoke_train(experiment, steps: int = 20) -> dict:
    """Tiny end-to-end train used by run_phase_checks --phase bsde."""
    model = experiment.model()
    bsde = experiment.bsde or {}
    net = ValueGradModel(model.I, hidden=tuple(bsde.get("hidden", (32, 32))),
                         seed=experiment.seed)
    cfg = TrainConfig(
        num_paths=int(bsde.get("smoke_paths", 64)),
        num_steps=int(bsde.get("smoke_steps", 16)),
        horizon=float(bsde.get("smoke_horizon", 0.5)),
        steps=steps, lr=float(bsde.get("lr", 1e-3)),
        **lr_schedule_kwargs(bsde),
        **bound_curriculum_kwargs(bsde),
        backend=experiment.reflection.get("backend", "numba"),
        reference_drift=bsde.get("reference_drift", None),
        loss_scale_by_horizon_sq=bool(bsde.get("loss_scale_by_horizon_sq", True)),
        val_interval=max(1, steps // 2), ckpt_interval=0,
        seed=experiment.seed,
    )
    trainer = BSDETrainer(model, net, cfg)
    history = trainer.train()
    return {"final_loss": history[-1]["loss"], "v0": net.value_at_origin()}
