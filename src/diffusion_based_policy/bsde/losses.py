"""BSDE / HJB loss in TensorFlow (math_foundation.md Sections 3.5, 4.2, 4.4).

The Hamiltonian/BSDE driver

    F(z, g) = htilde.z + min_x { g.R theta_x + c.K theta_x }

decomposes (for the single-resource paper networks) by resource block:

    min term = sum_{free blocks k} ( sum_{j in B_k} beta_j pi_j  -  block_max_k ),
    pi_j     = b ( g.R_.j + (K^T c_tilde)_j ),
    block_max_k = max(0, max_{feasible j} pi_j)  if idle allowed, else max pi_j.

The discrete residual on a reference path 0 = t_0 < ... < t_N = T is

    delta = e^{-gamma T} V(z_N) - V(z_0)
            - sum_m e^{-gamma t_m} G(z_m) . (sigma dW_m)
            + sum_m e^{-gamma t_m} { F(z_m, G(z_m)) - G(z_m).(m_ref - zeta) } dt,

and the BSDE loss is E[delta^2] / T^2 by default, the normalization used by
the short-segment BSDE training runs.
"""
from __future__ import annotations

import tensorflow as tf

from ..bcp import BCPModel


class HamiltonianTF:
    """Vectorized TF driver F(z, g) for a fixed network."""

    def __init__(self, model: BCPModel, dtype=tf.float32, tol: float = 1e-7):
        from ..validation import require_single_resource_per_activity
        require_single_resource_per_activity(model.spec)
        self.dtype = dtype
        self.default_bound = float(model.params.b)
        self.tol = tol
        self._neg = tf.constant(-1e30, dtype)        # typed sentinel for masked argmax
        npd = dtype.as_numpy_dtype
        self.R = tf.constant(model.R.astype(npd))                  # (I, J)
        self.offset = tf.constant(model.index_offset.astype(npd))  # (J,)
        self.h_tilde = tf.constant(model.h_tilde.astype(npd))      # (I,)
        self.beta = model.params.nominal_allocation.astype(npd)    # (J,)
        # consumption indicator (I, J)
        self.C_ind = tf.constant((model.spec.consumption > 0).astype(npd))
        # optimized blocks (skip no-rejection input rows)
        no_rej = set(int(k) for k in model.spec.no_rejection_resources)
        self.blocks = []
        for k in range(model.spec.num_resources):
            if k in no_rej:
                continue
            idx = model.spec.resource_block(k)
            if idx.size == 0:
                continue
            self.blocks.append({
                "idx": tf.constant(idx, tf.int32),
                "beta": tf.constant(self.beta[idx]),
                "idle": bool(model.spec.idle_allowed[k]),
                "is_processing": bool(model.spec.is_processing[k]),
            })

    def _bounds(self, control_bound):
        """Return (b_sched, b_reject). control_bound may be a scalar (both equal)
        or a (b_sched, b_reject) pair to give processing (scheduling) blocks and
        input (rejection/routing) blocks different control bounds."""
        if control_bound is None:
            b = float(self.default_bound)
            return b, b
        if isinstance(control_bound, (tuple, list)):
            return float(control_bound[0]), float(control_bound[1])
        b = float(control_bound)
        return b, b

    def policy_index(self, g, control_bound: float | None = None):
        # pi = b (g R + offset)   g: (B, I) -> (B, J)
        b_eff = self.default_bound if control_bound is None else float(control_bound)
        return b_eff * (tf.matmul(tf.cast(g, self.dtype), self.R) + self.offset)

    def min_term(self, z, g, control_bound: float | None = None):
        z = tf.cast(z, self.dtype)
        b_sched, b_reject = self._bounds(control_bound)
        s = tf.matmul(tf.cast(g, self.dtype), self.R) + self.offset   # (B, J), b=1 scores
        empty = tf.cast(z <= self.tol, self.dtype)        # (B, I)
        forbidden = tf.matmul(empty, self.C_ind) > 0.5    # (B, J) bool
        total = tf.zeros(tf.shape(z)[0], self.dtype)
        for blk in self.blocks:
            b_blk = b_sched if blk["is_processing"] else b_reject
            pib = b_blk * tf.gather(s, blk["idx"], axis=1)  # (B, |idx|)
            fb = tf.gather(forbidden, blk["idx"], axis=1)
            masked = tf.where(fb, self._neg, pib)
            best = tf.reduce_max(masked, axis=1)
            best = tf.where(best < -1e29, tf.zeros_like(best), best)
            block_max = tf.maximum(tf.constant(0.0, self.dtype), best) if blk["idle"] else best
            total += tf.linalg.matvec(pib, blk["beta"]) - block_max
        return total

    def driver_F(self, z, g, control_bound: float | None = None):
        z = tf.cast(z, self.dtype)
        return tf.linalg.matvec(z, self.h_tilde) + self.min_term(z, g, control_bound)


def bsde_loss(net, batch: dict, model: BCPModel, hamiltonian: HamiltonianTF | None = None,
              grad_nonneg_weight: float = 0.0,
              control_bound: float | None = None,
              scale_by_horizon_sq: bool = True):
    """Return (total_loss, components) for one reference-path batch."""
    dtype = net.dtype_
    npd = dtype.as_numpy_dtype
    H = hamiltonian or HamiltonianTF(model, dtype=dtype)

    z = tf.constant(batch["z"].astype(npd))          # (B, N+1, I)
    dw = tf.constant(batch["dw"].astype(npd))        # (B, N, I)
    disc = tf.constant(batch["disc"].astype(npd))    # (N+1,)
    dt = tf.constant(npd(batch["dt"]))
    rdz = tf.constant(batch["ref_drift_minus_zeta"].astype(npd))  # (I,)

    B = tf.shape(z)[0]
    N = batch["num_steps"]
    I = model.I

    z0 = z[:, 0, :]
    zN = z[:, N, :]
    V0 = net.value(z0)                                # (B,)
    VN = net.value(zN)                                # (B,)

    z_int = tf.reshape(z[:, :N, :], (-1, I))          # (B*N, I)
    G = net.grad(z_int)                               # (B*N, I)
    F = H.driver_F(z_int, G, control_bound=control_bound)  # (B*N,)
    F = tf.reshape(F, (B, N))
    G = tf.reshape(G, (B, N, I))

    stoch = tf.reduce_sum(G * dw, axis=-1)            # (B, N)
    drift = F - tf.reduce_sum(G * rdz, axis=-1)       # (B, N)

    disc_int = disc[:N]                               # (N,)
    sum_stoch = tf.reduce_sum(disc_int * stoch, axis=1)            # (B,)
    sum_drift = tf.reduce_sum(disc_int * drift, axis=1) * dt       # (B,)

    delta = disc[N] * VN - V0 - sum_stoch + sum_drift
    residual_sq_raw = tf.reduce_mean(tf.square(delta))
    scale = tf.square(tf.cast(batch["horizon"], dtype)) if scale_by_horizon_sq else tf.constant(1.0, dtype)
    residual_sq = residual_sq_raw / scale

    total = residual_sq
    penalty = tf.constant(0.0, dtype)
    if grad_nonneg_weight > 0.0:
        # linear hinge on negative gradients (Kasikaralar et al., Algorithm 3:
        # Lambda * sum max(-G, 0)); mean keeps it batch-size invariant, and it is
        # not horizon-rescaled since G itself is horizon-independent.
        penalty = grad_nonneg_weight * tf.reduce_mean(tf.nn.relu(-G))
        total = total + penalty

    components = {
        "residual_sq": residual_sq,
        "residual_sq_raw": residual_sq_raw,
        "penalty_grad_nonneg": penalty,
        "total": total,
        "v0_mean": tf.reduce_mean(V0),
        "delta_std": tf.math.reduce_std(delta),
    }
    return total, components
