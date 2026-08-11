"""Scaled batched Lemke reflection backend.

Solves the reflection LCP

    L >= 0,  Y = X + H L >= 0,  L_i Y_i = 0

as a standard LCP ``0 <= z ⟂ q + M z >= 0`` with ``M = H`` and
``q = X``.  The matrix is fixed for an experiment, so this backend equilibrates
it once with positive row/column scalings and then solves each batch with a
Numba-parallel Lemke pivot kernel.

This backend is intentionally separate from ``pgs_enum_lcp.py`` so PGS/basis enum
and Lemke can be compared directly.
"""
from __future__ import annotations

import numpy as np

from .base import ReflectionResult, lcp_residual

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def deco(f):
            return f
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return deco

    prange = range


def equilibrate_lcp_matrix(M: np.ndarray, n_iter: int = 10,
                           eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positive diagonal row/column scalings for an LCP matrix.

    The scaled LCP is

        u >= 0,  Dr q + Dr M Dc u >= 0,  u_i w_i = 0,

    and the original solution is ``z = Dc u``.  Positive diagonal scaling
    preserves the complementarity structure while improving pivot numerics.
    """
    M = np.asarray(M, dtype=np.float64)
    A = np.abs(M).copy()
    row_scale = np.ones(M.shape[0], dtype=np.float64)
    col_scale = np.ones(M.shape[1], dtype=np.float64)
    for _ in range(int(n_iter)):
        row_norm = np.maximum(np.linalg.norm(A, ord=np.inf, axis=1), eps)
        row_scale /= row_norm
        A = A / row_norm[:, None]

        col_norm = np.maximum(np.linalg.norm(A, ord=np.inf, axis=0), eps)
        col_scale /= col_norm
        A = A / col_norm[None, :]
    scaled = row_scale[:, None] * M * col_scale[None, :]
    return (np.ascontiguousarray(row_scale),
            np.ascontiguousarray(col_scale),
            np.ascontiguousarray(scaled))


@njit(cache=True)
def _complement(var: int, n: int) -> int:  # pragma: no cover - njit
    if var < n:
        return n + var
    return var - n


@njit(cache=True)
def _pivot(T, rhs, basis, row: int, entering: int):  # pragma: no cover - njit
    nrows, ncols = T.shape
    pivot = T[row, entering]
    inv = 1.0 / pivot
    for j in range(ncols):
        T[row, j] *= inv
    rhs[row] *= inv

    for i in range(nrows):
        if i == row:
            continue
        factor = T[i, entering]
        if factor != 0.0:
            for j in range(ncols):
                T[i, j] -= factor * T[row, j]
            rhs[i] -= factor * rhs[row]
    basis[row] = entering


@njit(cache=True)
def _solve_one_scaled_lemke(q, M, H_orig, row_scale, col_scale,
                            max_pivots: int, tol: float, pivot_tol: float,
                            residual_tol: float, Y, L):  # pragma: no cover - njit
    n = q.shape[0]
    nvars = 2 * n + 1
    z0 = 2 * n

    T = np.zeros((n, nvars), dtype=np.float64)
    rhs = np.empty(n, dtype=np.float64)
    basis = np.empty(n, dtype=np.int64)

    min_rhs = 1.0e300
    min_row = 0
    all_nonnegative = True
    for i in range(n):
        qi = row_scale[i] * q[i]
        rhs[i] = qi
        basis[i] = i
        T[i, i] = 1.0
        T[i, z0] = -1.0
        for j in range(n):
            T[i, n + j] = -M[i, j]
        if qi < -tol:
            all_nonnegative = False
        if qi < min_rhs:
            min_rhs = qi
            min_row = i

    if all_nonnegative:
        for i in range(n):
            L[i] = 0.0
            Y[i] = q[i]
        return True, 0

    leaving = basis[min_row]
    _pivot(T, rhs, basis, min_row, z0)
    entering = _complement(leaving, n)
    pivots = 1

    done = False
    ratio_tol = 1e-13
    for _ in range(max_pivots):
        best_row = -1
        best_ratio = 1.0e300
        best_label = nvars + 1
        for i in range(n):
            col = T[i, entering]
            if col > pivot_tol:
                val = rhs[i]
                if val < 0.0 and val > -tol:
                    val = 0.0
                ratio = val / col
                label = basis[i]
                if ratio < best_ratio - ratio_tol:
                    best_ratio = ratio
                    best_row = i
                    best_label = label
                elif ratio <= best_ratio + ratio_tol and label < best_label:
                    best_row = i
                    best_label = label

        if best_row < 0:
            return False, pivots

        leaving = basis[best_row]
        _pivot(T, rhs, basis, best_row, entering)
        pivots += 1

        for i in range(n):
            if rhs[i] < 0.0 and rhs[i] > -tol:
                rhs[i] = 0.0

        if leaving == z0:
            done = True
            break
        entering = _complement(leaving, n)

    if not done:
        return False, pivots

    u = np.zeros(n, dtype=np.float64)
    for i in range(n):
        var = basis[i]
        val = rhs[i]
        if val < 0.0 and val > -tol:
            val = 0.0
        if n <= var < 2 * n:
            u[var - n] = val

    for i in range(n):
        Li = col_scale[i] * u[i]
        if Li < 0.0 and Li > -tol:
            Li = 0.0
        L[i] = Li

    max_neg_y = 0.0
    max_neg_l = 0.0
    max_comp = 0.0
    max_abs_q = 0.0
    max_abs_l = 0.0
    max_row_sum = 0.0
    for i in range(n):
        yi = q[i]
        row_sum = 0.0
        for j in range(n):
            hij = H_orig[i, j]
            yi += hij * L[j]
            ah = hij if hij >= 0.0 else -hij
            row_sum += ah
        Y[i] = yi

        aq = q[i] if q[i] >= 0.0 else -q[i]
        if aq > max_abs_q:
            max_abs_q = aq
        al = L[i] if L[i] >= 0.0 else -L[i]
        if al > max_abs_l:
            max_abs_l = al
        if row_sum > max_row_sum:
            max_row_sum = row_sum

        if yi < 0.0 and -yi > max_neg_y:
            max_neg_y = -yi
        if L[i] < 0.0 and -L[i] > max_neg_l:
            max_neg_l = -L[i]
        comp = yi * L[i]
        if comp < 0.0:
            comp = -comp
        if comp > max_comp:
            max_comp = comp

    scale = 1.0 + max_abs_q + max_row_sum * max_abs_l
    good = (max_neg_y <= residual_tol * scale and
            max_neg_l <= residual_tol * scale and
            max_comp <= residual_tol * scale)
    return good, pivots


@njit(cache=True, parallel=True)
def _batched_scaled_lemke(X, H_orig, M_scaled, row_scale, col_scale,
                          max_pivots: int, tol: float, pivot_tol: float,
                          residual_tol: float):  # pragma: no cover - njit
    B, n = X.shape
    Y = np.empty_like(X)
    L = np.empty_like(X)
    ok = np.zeros(B, dtype=np.bool_)
    pivots = np.zeros(B, dtype=np.int64)
    for b in prange(B):
        good, n_pivots = _solve_one_scaled_lemke(
            X[b], M_scaled, H_orig, row_scale, col_scale,
            max_pivots, tol, pivot_tol, residual_tol, Y[b], L[b])
        ok[b] = good
        pivots[b] = n_pivots
    return Y, L, ok, pivots


class LemkeLCPReflection:
    """Reflection solver using scaled batched Lemke pivoting."""

    def __init__(self, H, tol: float = 1e-10, pivot_tol: float = 1e-12,
                 residual_tol: float = 1e-8, max_pivots: int | None = None,
                 scale: bool = True, scale_iterations: int = 10):
        self.H = np.ascontiguousarray(H, dtype=np.float64)
        self.dim = self.H.shape[0]
        self.tol = float(tol)
        self.pivot_tol = float(pivot_tol)
        self.residual_tol = float(residual_tol)
        self.max_pivots = int(max_pivots) if max_pivots is not None else max(1000, 50 * self.dim)
        self.scale = bool(scale)
        if self.scale:
            self.row_scale, self.col_scale, self.M_scaled = equilibrate_lcp_matrix(
                self.H, n_iter=scale_iterations)
        else:
            self.row_scale = np.ones(self.dim, dtype=np.float64)
            self.col_scale = np.ones(self.dim, dtype=np.float64)
            self.M_scaled = self.H.copy()
        self.row_scale = np.ascontiguousarray(self.row_scale, dtype=np.float64)
        self.col_scale = np.ascontiguousarray(self.col_scale, dtype=np.float64)
        self.M_scaled = np.ascontiguousarray(self.M_scaled, dtype=np.float64)
        self.last_ok = None
        self.last_pivots = None

    @property
    def backend(self) -> str:
        return "lemke" if _HAS_NUMBA else "lemke-python"

    def _solve_batch(self, X: np.ndarray):
        Y, L, ok, pivots = _batched_scaled_lemke(
            X, self.H, self.M_scaled, self.row_scale, self.col_scale,
            self.max_pivots, self.tol, self.pivot_tol, self.residual_tol)
        self.last_ok = ok
        self.last_pivots = pivots
        if not np.all(ok):
            failed = np.where(~ok)[0]
            first = int(failed[0])
            raise RuntimeError(
                f"Lemke LCP failed for {failed.size}/{X.shape[0]} samples; "
                f"first={first}, pivots={int(pivots[first])}")
        return Y, L, pivots

    def solve(self, X: np.ndarray, tighten: bool = True) -> ReflectionResult:
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y, L, pivots = self._solve_batch(X)
        res = lcp_residual(X, Y, self.H)
        return ReflectionResult(reflected=Y, boundary_push=L,
                                lcp_residual=res,
                                iterations=int(pivots.max(initial=0)),
                                fallback_count=0)

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = X.copy()
        need = np.where(~np.all(X >= 0.0, axis=1))[0]
        if need.size == 0:
            self.last_ok = np.ones(X.shape[0], dtype=bool)
            self.last_pivots = np.zeros(X.shape[0], dtype=np.int64)
            return Y
        Ys, _Ls, _pivots = self._solve_batch(np.ascontiguousarray(X[need]))
        Y[need] = Ys
        return Y
