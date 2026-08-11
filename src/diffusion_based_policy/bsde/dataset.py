"""BSDE datasets: deterministic reference-path batches for training/validation.

A batch is a plain dict of numpy arrays so the trainer (TensorFlow) and any
numpy reference checks can both consume it:

    z       (B, N+1, I)   reflected reference states at each time node
    dw      (B, N, I)     diffusion increments sigma dW_m
    disc    (N+1,)        e^{-gamma t_m}, t_m = m dt
    dt      float
    ref_drift_minus_zeta  (I,)  m - zeta  (the controlled part of the reference drift)

Builders are deterministic in the seed; ``make_validation_batch`` pins the seed
so the evaluation set is fixed across runs.
"""
from __future__ import annotations

import numpy as np

from ..bcp import BCPModel
from ..sim.diffusion import (
    ContinuousReferenceSampler,
    OnPolicyContinuousSampler,
    simulate_reference_paths,
)


def discount_nodes(gamma: float, dt: float, num_steps: int) -> np.ndarray:
    t = np.arange(num_steps + 1) * dt
    return np.exp(-gamma * t)


def _assemble(model: BCPModel, paths: dict) -> dict:
    dt = paths["dt"]
    disc = discount_nodes(model.gamma, dt, paths["num_steps"])
    ref_drift = paths["reference_drift"]
    return {
        "z": paths["z_path"].astype(np.float64),
        "dw": paths["dw"].astype(np.float64),
        "l_incr": paths["l_incr"].astype(np.float64),
        "disc": disc,
        "dt": dt,
        "horizon": paths["horizon"],
        "num_steps": paths["num_steps"],
        "ref_drift_minus_zeta": (ref_drift - model.zeta).astype(np.float64),
        "reference_drift": ref_drift.astype(np.float64),
    }


def make_training_batch(
    model: BCPModel,
    num_paths: int,
    num_steps: int,
    horizon: float,
    seed: int,
    backend: str = "numba",
    reference_drift=None,
    init: str = "zeros",
    init_scale: float = 1.0,
) -> dict:
    paths = simulate_reference_paths(
        model, num_paths=num_paths, num_steps=num_steps, horizon=horizon,
        seed=seed, backend=backend, reference_drift=reference_drift,
        init=init, init_scale=init_scale,
    )
    return _assemble(model, paths)


def make_validation_batch(
    model: BCPModel,
    num_paths: int,
    num_steps: int,
    horizon: float,
    seed: int = 999_983,
    backend: str = "numba",
    reference_drift=None,
    init: str = "zeros",
    init_scale: float = 1.0,
) -> dict:
    """Deterministic validation batch (fixed seed by default)."""
    return make_training_batch(
        model, num_paths=num_paths, num_steps=num_steps, horizon=horizon,
        seed=seed, backend=backend, reference_drift=reference_drift,
        init=init, init_scale=init_scale,
    )


def batch_stream(
    model: BCPModel,
    num_paths: int,
    num_steps: int,
    horizon: float,
    base_seed: int = 0,
    backend: str = "numba",
    reference_drift=None,
    init: str = "zeros",
    init_scale: float = 1.0,
):
    """Infinite generator of deterministic training batches (seed = base_seed + i)."""
    i = 0
    while True:
        yield make_training_batch(
            model, num_paths=num_paths, num_steps=num_steps, horizon=horizon,
            seed=base_seed + i, backend=backend, reference_drift=reference_drift,
            init=init, init_scale=init_scale,
        )
        i += 1


class ContinuousBatchSampler:
    """Assembled-batch view of the stateful continuous reference-path sampler.

    Wraps ``sim.ContinuousReferenceSampler`` and returns batches in the same dict
    format the loss consumes (z / dw / disc / ref_drift_minus_zeta). The path is
    continuous across calls: ``batch()`` returns the next segment continuing from
    where the previous one ended, after an initial ``warmup_segments`` burn-in.
    """

    def __init__(self, model: BCPModel, num_paths: int, num_steps: int,
                 horizon: float, backend: str = "numba", reference_drift=None,
                 init: str = "zeros", init_scale: float = 1.0, seed: int = 0,
                 warmup_segments: int = 0):
        self.model = model
        self._sampler = ContinuousReferenceSampler(
            model, num_paths=num_paths, num_steps=num_steps, horizon=horizon,
            backend=backend, reference_drift=reference_drift, init=init,
            init_scale=init_scale, seed=seed, warmup_segments=warmup_segments)

    @property
    def segments_done(self) -> int:
        return self._sampler.segments_done

    def batch(self) -> dict:
        return _assemble(self.model, self._sampler.advance())


class PooledContinuousSampler:
    """Bounded reflected-path pool, reused across epochs (the published scheme).

    Generates ``pool_segments`` continuous segments ONCE (after a warmup), stores
    them, and serves them by a per-epoch-shuffled index. This caps the (expensive)
    reflection work at ``pool_segments`` instead of one fresh segment per training
    step -- a fixed segment pool reused across all epochs keeps sampling fast.

    Memory: ~ pool_segments * num_paths * num_steps * I * 8 bytes (float64) for the
    state pool plus a similar amount for the increments.
    """

    def __init__(self, model: BCPModel, num_paths: int, num_steps: int,
                 horizon: float, pool_segments: int, backend: str = "numba",
                 reference_drift=None, init: str = "zeros", init_scale: float = 1.0,
                 seed: int = 0, warmup_segments: int = 0, shuffle_seed: int = 20260420):
        self.model = model
        self.pool_segments = int(pool_segments)
        cont = ContinuousReferenceSampler(
            model, num_paths=num_paths, num_steps=num_steps, horizon=horizon,
            backend=backend, reference_drift=reference_drift, init=init,
            init_scale=init_scale, seed=seed, warmup_segments=warmup_segments)
        I = model.I
        self._z = np.empty((self.pool_segments, num_paths, num_steps + 1, I), np.float64)
        self._dw = np.empty((self.pool_segments, num_paths, num_steps, I), np.float64)
        for i in range(self.pool_segments):
            seg = cont.advance()
            self._z[i] = seg["z_path"]
            self._dw[i] = seg["dw"]
        m = cont.m
        self._meta = {
            "dt": cont.dt,
            "disc": discount_nodes(model.gamma, cont.dt, num_steps),
            "num_steps": num_steps,
            "horizon": float(horizon),
            "ref_drift_minus_zeta": (m - model.zeta).astype(np.float64),
            "reference_drift": m.astype(np.float64),
        }
        self._rng = np.random.default_rng(shuffle_seed)
        self._order = np.arange(self.pool_segments)
        self._rng.shuffle(self._order)
        self._pos = 0
        self.epochs_done = 0

    def batch(self) -> dict:
        if self._pos >= self.pool_segments:           # exhausted -> next epoch
            self._rng.shuffle(self._order)
            self._pos = 0
            self.epochs_done += 1
        idx = int(self._order[self._pos])
        self._pos += 1
        b = dict(self._meta)
        b["z"] = self._z[idx].astype(np.float64)
        b["dw"] = self._dw[idx].astype(np.float64)
        b["l_incr"] = np.zeros_like(b["dw"])
        return b


class PooledOnPolicySampler:
    """Bounded reflected-path pool generated under a FROZEN behaviour policy.

    Like ``PooledContinuousSampler`` but the segment drift is the behaviour
    policy's ``mu(z) = zeta + theta(z) @ R.T`` (one fixed pool, reused across
    epochs at offline speed). Each pooled segment stores the per-step reference
    drift correction so the loss subtracts the state-dependent ``m - zeta``.
    This is the cheap "lagged / fitted" on-policy: simulate once per stage under
    the previous stage's policy, then train on the fixed pool.
    """

    def __init__(self, model: BCPModel, num_paths: int, num_steps: int,
                 horizon: float, pool_segments: int, grad_fn, behavior_bound: float,
                 backend: str = "numba", init: str = "zeros", init_scale: float = 1.0,
                 seed: int = 0, warmup_segments: int = 0, shuffle_seed: int = 20260420):
        self.model = model
        self.pool_segments = int(pool_segments)
        cont = OnPolicyContinuousSampler(
            model, num_paths=num_paths, num_steps=num_steps, horizon=horizon,
            grad_fn=grad_fn, behavior_bound=behavior_bound, backend=backend,
            init=init, init_scale=init_scale, seed=seed, warmup_segments=warmup_segments)
        I = model.I
        self._z = np.empty((self.pool_segments, num_paths, num_steps + 1, I), np.float64)
        self._dw = np.empty((self.pool_segments, num_paths, num_steps, I), np.float64)
        self._rdz = np.empty((self.pool_segments, num_paths, num_steps, I), np.float64)
        for i in range(self.pool_segments):
            seg = cont.advance()
            self._z[i] = seg["z_path"]
            self._dw[i] = seg["dw"]
            self._rdz[i] = seg["rdz"]
        self._meta = {
            "dt": cont.dt,
            "disc": discount_nodes(model.gamma, cont.dt, num_steps),
            "num_steps": num_steps,
            "horizon": float(horizon),
        }
        self._rng = np.random.default_rng(shuffle_seed)
        self._order = np.arange(self.pool_segments)
        self._rng.shuffle(self._order)
        self._pos = 0
        self.epochs_done = 0

    def batch(self) -> dict:
        if self._pos >= self.pool_segments:
            self._rng.shuffle(self._order)
            self._pos = 0
            self.epochs_done += 1
        idx = int(self._order[self._pos])
        self._pos += 1
        b = dict(self._meta)
        b["z"] = self._z[idx].astype(np.float64)
        b["dw"] = self._dw[idx].astype(np.float64)
        b["l_incr"] = np.zeros_like(b["dw"])
        b["ref_drift_minus_zeta"] = self._rdz[idx].astype(np.float64)   # (P, N, I)
        return b


class LiveHybridPool:
    """Refreshable on-policy + offline hybrid pool for TRUE on-policy training.

    Holds two chained sub-samplers: ``n_on`` paths driven by the LIVE policy
    (drift ``zeta + theta(z) R^T`` with theta from a grad_fn that wraps the net
    being trained) and ``n_off`` paths under the constant offline reference drift
    (an anchor that keeps the origin / small states covered). ``refresh(grad_fn)``
    regenerates the segment pool under the *current* network -- so as the policy
    updates, the reference co-adapts and stays in distribution (the chained
    on-policy paths explore the operating region the policy actually visits).
    Between refreshes the pool is reused (shuffled), like the offline pool.
    """

    def __init__(self, model: BCPModel, num_paths: int, num_steps: int,
                 horizon: float, pool_segments: int, behavior_bound: float,
                 behavior_ratio: float = 0.5, reference_drift=None,
                 backend: str = "numba", init: str = "zeros", init_scale: float = 1.0,
                 seed: int = 0, warmup_segments: int = 200, shuffle_seed: int = 20260420):
        self.model = model
        self.num_steps = num_steps
        self.pool_segments = int(pool_segments)
        self.warmup_segments = int(warmup_segments)
        n_on = max(1, min(num_paths, int(round(num_paths * behavior_ratio))))
        n_off = num_paths - n_on
        self.n_on, self.n_off = n_on, n_off
        I = model.I
        self._on = OnPolicyContinuousSampler(
            model, num_paths=n_on, num_steps=num_steps, horizon=horizon,
            grad_fn=None, behavior_bound=behavior_bound, backend=backend,
            init=init, init_scale=init_scale, seed=seed, warmup_segments=0, chain=True)
        self._off = (ContinuousReferenceSampler(
            model, num_paths=n_off, num_steps=num_steps, horizon=horizon,
            backend=backend, reference_drift=reference_drift, init=init,
            init_scale=init_scale, seed=seed + 7919, warmup_segments=warmup_segments)
            if n_off > 0 else None)
        m_off = (model.zeta.copy() if reference_drift is None
                 else np.asarray(reference_drift, float))
        self._rdz_off = (m_off - model.zeta).astype(np.float64)
        self.dt = horizon / num_steps
        self._meta = {
            "dt": self.dt, "disc": discount_nodes(model.gamma, self.dt, num_steps),
            "num_steps": num_steps, "horizon": float(horizon),
        }
        self._z = np.empty((self.pool_segments, num_paths, num_steps + 1, I), np.float64)
        self._dw = np.empty((self.pool_segments, num_paths, num_steps, I), np.float64)
        self._rdz = np.empty((self.pool_segments, num_paths, num_steps, I), np.float64)
        self._rng = np.random.default_rng(shuffle_seed)
        self._order = np.arange(self.pool_segments)
        self._pos = self.pool_segments    # empty until first refresh
        self._warmed = False
        self.refreshes = 0

    def refresh(self, grad_fn, behavior_bound: float | None = None) -> None:
        """Regenerate the pool under the current live policy ``grad_fn``."""
        self._on.grad_fn = grad_fn
        if behavior_bound is not None:
            self._on.a = float(behavior_bound)
        if not self._warmed:        # bring the on-policy chain onto its stationary set
            for _ in range(self.warmup_segments):
                self._on.advance()
            self._warmed = True
        for i in range(self.pool_segments):
            so = self._on.advance()
            z, dw, rdz = so["z_path"], so["dw"], so["rdz"]
            if self._off is not None:
                sf = self._off.advance()
                rdz_off = np.broadcast_to(self._rdz_off, (self.n_off, self.num_steps, z.shape[-1]))
                z = np.concatenate([z, sf["z_path"]], axis=0)
                dw = np.concatenate([dw, sf["dw"]], axis=0)
                rdz = np.concatenate([rdz, rdz_off], axis=0)
            self._z[i] = z
            self._dw[i] = dw
            self._rdz[i] = rdz
        self._rng.shuffle(self._order)
        self._pos = 0
        self.refreshes += 1

    def batch(self) -> dict:
        if self._pos >= self.pool_segments:
            self._rng.shuffle(self._order)
            self._pos = 0
        idx = int(self._order[self._pos])
        self._pos += 1
        b = dict(self._meta)
        b["z"] = self._z[idx].astype(np.float64)
        b["dw"] = self._dw[idx].astype(np.float64)
        b["l_incr"] = np.zeros_like(b["dw"])
        b["ref_drift_minus_zeta"] = self._rdz[idx].astype(np.float64)
        return b


def make_tf_dataset(
    model: BCPModel,
    num_paths: int,
    num_steps: int,
    horizon: float,
    base_seed: int = 0,
    backend: str = "numba",
    reference_drift=None,
    init: str = "zeros",
    init_scale: float = 1.0,
    prefetch: int = 2,
):
    """Optional tf.data.Dataset wrapping the deterministic batch stream.

    Only the z / dw tensors are streamed; scalar metadata (dt, disc, ref drift)
    is constant per experiment and supplied to the loss separately.
    """
    import tensorflow as tf

    I = model.I

    def gen():
        for batch in batch_stream(model, num_paths, num_steps, horizon,
                                  base_seed=base_seed, backend=backend,
                                  reference_drift=reference_drift,
                                  init=init, init_scale=init_scale):
            yield (batch["z"].astype(np.float32), batch["dw"].astype(np.float32))

    sig = (
        tf.TensorSpec(shape=(num_paths, num_steps + 1, I), dtype=tf.float32),
        tf.TensorSpec(shape=(num_paths, num_steps, I), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    return ds.prefetch(prefetch)


# --------------------------------------------------------------------------- #
# Optional GPU-fused sampler: Euler + basis-enum reflection inside one
# tf.function. Internal FP64 for correct basis selection; output cast to dtype.
# --------------------------------------------------------------------------- #
def make_fused_reference_sampler(model: BCPModel, num_steps: int, horizon: float,
                                 dtype="float32", jit_compile: bool = False):
    import tensorflow as tf

    from ..reflection.gpu_lcp import _precompute_basis_tensors

    I = model.I
    dt = float(horizon) / num_steps
    sqrt_dt = float(np.sqrt(dt))
    sigma = np.asarray(model.sigma, np.float64)
    H = np.asarray(model.H, np.float64)
    m = np.asarray(model.zeta, np.float64)

    B_mask, A, HA, valid = _precompute_basis_tensors(H, I)
    f64 = tf.float64
    B_mask_t = tf.constant(B_mask, f64)
    A_t = tf.constant(A, f64)
    HA_t = tf.constant(HA, f64)
    valid_t = tf.constant(valid, tf.bool)
    sigma_t = tf.constant(sigma, f64)
    m_t = tf.constant(m, f64)
    dt_t = tf.constant(dt, f64)
    sqrt_dt_t = tf.constant(sqrt_dt, f64)
    tol_t = tf.constant(1e-10, f64)
    out_dtype = tf.dtypes.as_dtype(dtype)

    def basis_enum(X):
        L_all = -tf.einsum("mij,bj->mbi", A_t, X)
        W_all = tf.expand_dims(X, 0) - tf.einsum("mij,bj->mbi", HA_t, X)
        B_exp = tf.expand_dims(B_mask_t, 1)
        N_exp = 1.0 - B_exp
        feas = (tf.reduce_min(L_all * B_exp + N_exp, -1) >= -tol_t) & \
               (tf.reduce_min(W_all * N_exp + B_exp, -1) >= -tol_t)
        feas = feas & tf.expand_dims(valid_t, 1)
        first = tf.argmax(tf.cast(feas, tf.int32), axis=0, output_type=tf.int32)
        bsz = tf.shape(X)[0]
        idx = tf.stack([first, tf.range(bsz, dtype=tf.int32)], axis=1)
        W = tf.gather_nd(W_all, idx)
        return tf.maximum(W * (1.0 - tf.gather(B_mask_t, first)), 0.0)

    @tf.function(jit_compile=jit_compile, reduce_retracing=True)
    def fused(z0, seed):
        z = tf.cast(z0, f64)
        zs = tf.TensorArray(f64, size=num_steps + 1)
        dws = tf.TensorArray(f64, size=num_steps)
        zs = zs.write(0, z)
        bsz = tf.shape(z)[0]
        for k in tf.range(num_steps):
            raw = tf.random.stateless_normal([bsz, I], seed=seed + k * 1_000_003, dtype=f64)
            dw = tf.matmul(raw, sigma_t, transpose_b=True) * sqrt_dt_t
            x = z + m_t * dt_t + dw
            z = basis_enum(x)
            zs = zs.write(k + 1, z)
            dws = dws.write(k, dw)
        z_path = tf.transpose(zs.stack(), [1, 0, 2])     # (B, N+1, I)
        dw_path = tf.transpose(dws.stack(), [1, 0, 2])   # (B, N, I)
        return tf.cast(z_path, out_dtype), tf.cast(dw_path, out_dtype)

    return fused
