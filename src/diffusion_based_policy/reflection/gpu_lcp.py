"""GPU reflection backend: exact batched basis enumeration in TensorFlow.

Enumerates all 2^dim active sets of the LCP  Y = X + H L,  Y,L >= 0,  Y^T L = 0
and selects, per sample, the first feasible basis. Internal precision is FP64 to
avoid basis-selection errors; the public API returns FP64 numpy arrays.

Runs on GPU when one is visible, otherwise on CPU under the same TensorFlow code
path (still exact, just slower), which makes it a valid cross-check of the numba
backend everywhere.
"""
from __future__ import annotations

import numpy as np

from .base import ReflectionResult, lcp_residual


def _precompute_basis_tensors(H: np.ndarray, dim: int):
    num = 1 << dim
    B_mask = np.zeros((num, dim))
    A = np.zeros((num, dim, dim))
    valid = np.zeros(num, dtype=bool)
    for m in range(num):
        B_idx = [i for i in range(dim) if m & (1 << i)]
        for i in B_idx:
            B_mask[m, i] = 1.0
        if not B_idx:
            valid[m] = True
            continue
        B = np.array(B_idx, dtype=np.intp)
        try:
            inv = np.linalg.inv(H[np.ix_(B, B)])
        except np.linalg.LinAlgError:
            continue
        valid[m] = True
        for ii, i in enumerate(B_idx):
            for jj, j in enumerate(B_idx):
                A[m, i, j] = inv[ii, jj]
    HA = np.einsum("ik,mkj->mij", H, A)
    return B_mask, A, HA, valid


class GPUBasisReflection:
    """Reflection solver via TensorFlow basis enumeration."""

    def __init__(self, H, tol: float = 1e-10, jit_compile: bool = False):
        import tensorflow as tf

        self._tf = tf
        self.H = np.ascontiguousarray(H, dtype=np.float64)
        self.dim = self.H.shape[0]
        self.tol = float(tol)
        B_mask, A, HA, valid = _precompute_basis_tensors(self.H, self.dim)
        dt = tf.float64
        self._B_mask = tf.constant(B_mask, dtype=dt)
        self._A = tf.constant(A, dtype=dt)
        self._HA = tf.constant(HA, dtype=dt)
        self._valid = tf.constant(valid, dtype=tf.bool)
        self._tol_tf = tf.constant(self.tol, dtype=dt)
        self._H_tf = tf.constant(self.H, dtype=dt)

        @tf.function(jit_compile=jit_compile, reduce_retracing=True)
        def _project(X):  # X: (B, dim) fp64
            L_all = -tf.einsum("mij,bj->mbi", self._A, X)
            W_all = tf.expand_dims(X, 0) - tf.einsum("mij,bj->mbi", self._HA, X)
            B_exp = tf.expand_dims(self._B_mask, 1)
            N_exp = 1.0 - B_exp
            L_check = L_all * B_exp + N_exp
            W_check = W_all * N_exp + B_exp
            feasible = (tf.reduce_min(L_check, axis=-1) >= -self._tol_tf) & \
                       (tf.reduce_min(W_check, axis=-1) >= -self._tol_tf)
            feasible = feasible & tf.expand_dims(self._valid, 1)
            first = tf.argmax(tf.cast(feasible, tf.int32), axis=0, output_type=tf.int32)
            bsz = tf.shape(X)[0]
            idx = tf.stack([first, tf.range(bsz, dtype=tf.int32)], axis=1)
            W = tf.gather_nd(W_all, idx)
            L = tf.gather_nd(L_all, idx)
            basis_mask = tf.gather(self._B_mask, first)
            W = W * (1.0 - basis_mask)            # zero reflected state on active set
            L = L * basis_mask                    # local time only on active set
            return tf.maximum(W, 0.0), tf.maximum(L, 0.0)

        self._project_tf = _project

    @property
    def backend(self) -> str:
        return "gpu"

    def solve(self, X: np.ndarray) -> ReflectionResult:
        tf = self._tf
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y_tf, L_tf = self._project_tf(tf.constant(X, dtype=tf.float64))
        Y = Y_tf.numpy()
        L = L_tf.numpy()
        res = lcp_residual(X, Y, self.H)
        return ReflectionResult(reflected=Y, boundary_push=L, lcp_residual=res)

    def project(self, X: np.ndarray) -> np.ndarray:
        return self.solve(X).reflected


def gpu_available() -> bool:
    try:
        import tensorflow as tf
    except Exception:
        return False
    try:
        return len(tf.config.list_physical_devices("GPU")) > 0
    except Exception:
        return False
