"""TensorFlow value/gradient networks for the BSDE solver.

Implemented with ``tf.Module`` and explicit ``tf.Variable`` weights -- no
``tf.keras.Model``, ``tf.keras.layers``, ``model.compile`` or ``model.fit``. Two
separate networks (math_foundation.md Section 4.4):

    V_eta(z)  : R^I -> R     (value, used only at path endpoints)
    G_phi(z)  : R^I -> R^I   (gradient, used at every interior node)
"""
from __future__ import annotations

import os

import numpy as np
import tensorflow as tf


def _default_dtype():
    """Network dtype, overridable via BCP_NET_DTYPE (float32 | float64).

    Defaults to float32 (fast GPU compute). Set BCP_NET_DTYPE=float64 to run the
    value/gradient networks in double precision -- this removes the float32
    cancellation in the residual disc[N]*V(z_N) - V(z_0) when V is large.
    """
    return (tf.float64 if os.environ.get("BCP_NET_DTYPE", "float32").lower()
            in ("float64", "fp64", "64") else tf.float32)


class MLP(tf.Module):
    """Plain ELU multilayer perceptron with explicit tf.Variable weights."""

    def __init__(self, in_dim: int, out_dim: int, hidden=(100, 100, 100),
                 seed: int = 0, dtype=None, name=None):
        super().__init__(name=name)
        if dtype is None:
            dtype = _default_dtype()
        rng = np.random.default_rng(seed)
        self.dtype_ = dtype
        self.Ws: list[tf.Variable] = []
        self.bs: list[tf.Variable] = []
        dims = [in_dim, *hidden, out_dim]
        for i in range(len(dims) - 1):
            # Kaiming/He init (matched to the ELU rectifier); TF's built-in
            # initializer, seeded per layer for reproducibility.
            he = tf.keras.initializers.HeNormal(seed=int(rng.integers(2**31 - 1)))
            self.Ws.append(tf.Variable(he((dims[i], dims[i + 1]), dtype=dtype), name="w"))
            self.bs.append(tf.Variable(tf.zeros(dims[i + 1], dtype=dtype), name="b"))

    @tf.function(reduce_retracing=True)
    def __call__(self, z):
        h = tf.cast(z, self.dtype_)
        n = len(self.Ws)
        for i in range(n):
            h = tf.matmul(h, self.Ws[i]) + self.bs[i]
            if i < n - 1:
                h = tf.nn.elu(h)
        return h


class ValueGradModel(tf.Module):
    """Bundles the value network V_eta and gradient network G_phi."""

    def __init__(self, dim: int, hidden=(100, 100, 100), seed: int = 0,
                 dtype=None, name="value_grad"):
        super().__init__(name=name)
        if dtype is None:
            dtype = _default_dtype()
        self.dim = dim
        self.dtype_ = dtype
        self.value_net = MLP(dim, 1, hidden=hidden, seed=seed, dtype=dtype, name="V")
        self.grad_net = MLP(dim, dim, hidden=hidden, seed=seed + 1, dtype=dtype, name="G")

    @tf.function(reduce_retracing=True)
    def value(self, z):
        return tf.squeeze(self.value_net(z), axis=-1)   # (B,)

    @tf.function(reduce_retracing=True)
    def grad(self, z):
        return self.grad_net(z)                          # (B, I)

    @property
    def trainable_variables(self):  # type: ignore[override]
        return self.value_net.trainable_variables + self.grad_net.trainable_variables

    def value_at_origin(self) -> float:
        z0 = tf.zeros((1, self.dim), self.dtype_)
        return float(self.value(z0).numpy()[0])
