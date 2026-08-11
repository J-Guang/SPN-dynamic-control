"""Load a trained BSDE checkpoint and compute diagnostics."""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from ..bcp import BCPModel
from .dataset import make_validation_batch
from .losses import bsde_loss
from .model import ValueGradModel, _default_dtype


def load_model_from_checkpoint(ckpt_dir: str, dim: int, hidden=(100, 100, 100),
                               dtype=None) -> ValueGradModel:
    if dtype is None:
        dtype = _default_dtype()
    net = ValueGradModel(dim, hidden=tuple(hidden), dtype=dtype)
    ckpt = tf.train.Checkpoint(model=net)
    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint in {ckpt_dir}")
    # expect_partial: optimizer slots are not restored for evaluation.
    ckpt.restore(latest).expect_partial()
    return net


class BSDEEvaluator:
    def __init__(self, model: BCPModel, net: ValueGradModel):
        self.model = model
        self.net = net

    def value_at_origin(self) -> float:
        return self.net.value_at_origin()

    def validation_residual(self, num_paths=2048, num_steps=64, horizon=3.0,
                            backend="numba", reference_drift=None) -> dict:
        batch = make_validation_batch(
            self.model, num_paths=num_paths, num_steps=num_steps,
            horizon=horizon, backend=backend, reference_drift=reference_drift)
        _, comps = bsde_loss(self.net, batch, self.model)
        return {k: float(v.numpy()) for k, v in comps.items()}

    def brownian_value_mc(self, num_paths: int = 2048, horizon: float = 3.0,
                          dt: float | None = None, reference_drift=None,
                          control_bound: float | None = None,
                          backend: str = "numba", seed: int = 20260614) -> dict:
        """BCP Brownian value by simulation rollout (the paper's "BCP V_MC").

        Monte-Carlo estimate of V(0) by integrating the BSDE driver along the
        reference reflected diffusion:

            V_MC = E[ int_0^T e^{-gamma t} { F(Z, G(Z)) - G(Z).(m - zeta) } dt ],

        where Z is the reference diffusion (drift m, reflected by H) and the
        e^{-gamma T} V(Z_T) terminal is dropped (negligible at gamma T = 12).
        ``control_bound=None`` uses the full trained bound b.
        """
        from ..reflection import get_solver
        from .losses import HamiltonianTF

        model = self.model
        I = model.I
        dt = (0.01 / 64.0) if dt is None else float(dt)
        nsteps = int(round(horizon / dt))
        sigma = model.sigma
        m = (model.zeta.copy() if reference_drift is None
             else np.asarray(reference_drift, float))
        rdz = m - model.zeta
        gamma = model.gamma
        solver = get_solver(backend, model.H)
        ham = HamiltonianTF(model, dtype=self.net.dtype_)
        npd = self.net.dtype_.as_numpy_dtype

        @tf.function(reduce_retracing=True)
        def _grad_and_driver(z):
            g = self.net.grad(z)
            F = ham.driver_F(z, g, control_bound=control_bound)
            return g, F

        rng = np.random.default_rng(seed)
        z = np.zeros((num_paths, I))
        acc = np.zeros(num_paths)
        disc = 1.0
        sqrt_dt = np.sqrt(dt)
        for _ in range(nsteps):
            g, F = _grad_and_driver(tf.constant(z.astype(npd)))
            drift_term = F.numpy() - g.numpy() @ rdz       # F - G.(m - zeta)
            acc += disc * drift_term * dt
            disc *= np.exp(-gamma * dt)
            dw = (rng.standard_normal((num_paths, I)) * sqrt_dt) @ sigma.T
            z = solver.project(z + m * dt + dw)

        return {
            "v_mc": float(acc.mean()),
            "stderr": float(acc.std(ddof=1) / np.sqrt(num_paths)) if num_paths > 1 else 0.0,
            "num_paths": num_paths, "horizon": horizon, "dt": dt, "nsteps": nsteps,
            "reference_drift": m.tolist(),
            "control_bound": control_bound if control_bound is not None else model.params.b,
        }

    def gradient_at(self, z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=self.net.dtype_.as_numpy_dtype))
        return self.net.grad(tf.constant(z)).numpy()

    def gradient_fn(self):
        """Return g = G_phi(z) callable for the policy / prelimit simulator."""
        npd = self.net.dtype_.as_numpy_dtype

        def fn(z):
            z2 = np.atleast_2d(np.asarray(z, dtype=npd))
            g = self.net.grad(tf.constant(z2)).numpy()
            return g[0] if np.ndim(z) == 1 else g

        return fn

    def diagnostics(self, **kwargs) -> dict:
        out = {"v0": self.value_at_origin()}
        out.update({f"val_{k}": v for k, v in self.validation_residual(**kwargs).items()})
        return out
