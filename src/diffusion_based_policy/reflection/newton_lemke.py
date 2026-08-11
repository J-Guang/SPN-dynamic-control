"""Active-set / semismooth-Newton reflection backend with an exact Lemke fallback.

Solves the reflection LCP

    L >= 0,   Y = X + H L >= 0,   L_i Y_i = 0

for a fixed reflection matrix ``H`` over a large batch of states ``X``.

Why this backend exists
-----------------------
``pgs_enum_lcp`` (PGS + 2^dim basis enumeration) is fast only when either the
batch is interior-light (PGS converges) or ``dim`` is small enough to enumerate
2^dim bases. For the bigstep network ``H`` is a P-matrix but **not** an H-matrix
(it is not generalized-diagonally-dominant), so projected Gauss-Seidel does not
converge, and ``dim=8`` is already past the comfortable enumeration range -- so
``project()`` there falls back to a *serial Python* 2^dim enumeration loop and
the whole pool build collapses to a single core.

This backend replaces both with a method that is robust for any P-matrix and
scales polynomially in ``dim``:

* **Primary -- block principal pivoting (BPP).** This is the active-set form of
  the min-map semismooth Newton for the LCP: each iteration fixes a candidate
  partition (free set F where ``L>0, Y=0``; tight set T where ``L=0, Y>=0``),
  solves the small dense system ``H_FF L_F = -X_F`` for the free block, then
  flips the indices that came out infeasible. For a P-matrix every principal
  submatrix ``H_FF`` is nonsingular, so each step is well defined; it typically
  converges in a handful of steps. It is "guided basis search" -- the scalable
  replacement for enumerating all 2^dim bases.

* **Fallback -- scaled Lemke pivoting.** Block pivoting can cycle on degenerate
  states. The rare path that does not reach feasibility within ``bpp_max_iter``
  escalates -- *inside the same numba kernel* -- to the exact scaled-Lemke pivot
  solve from :mod:`lemke_lcp`, which is finitely terminating for P-matrices.

Crucially the whole per-path solve (BPP *and* its Lemke escalation) runs inside
one ``@njit(parallel=True)`` ``prange`` over the batch, so no path ever drops to
a serial Python loop: parallelism is across independent LCPs, while each LCP is
solved by its own (sequential) method on a worker thread.

Kept deliberately separate from ``pgs_enum_lcp`` / ``lemke_lcp`` so PGS+enum, pure
Lemke and Newton+Lemke can be benchmarked head to head.
"""
from __future__ import annotations

import numpy as np

from .base import ReflectionResult, lcp_residual
from .lemke_lcp import _solve_one_scaled_lemke, equilibrate_lcp_matrix

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


@njit(cache=True)
def _solve_small(A, b, n):  # pragma: no cover - njit
    """In-place Gaussian elimination with partial pivoting.

    Solves ``A[:n, :n] x = b[:n]`` and writes the solution back into ``b[:n]``.
    Returns ``False`` if a (near-)singular pivot is hit, so the caller can
    escalate to the exact pivot fallback instead of trusting a garbage solve.
    Only the top-left ``n x n`` block of ``A`` and ``b[:n]`` are touched.
    """
    for k in range(n):
        p = k
        maxv = abs(A[k, k])
        for i in range(k + 1, n):
            v = abs(A[i, k])
            if v > maxv:
                maxv = v
                p = i
        if maxv < 1e-14:
            return False
        if p != k:
            for j in range(n):
                t = A[k, j]
                A[k, j] = A[p, j]
                A[p, j] = t
            t = b[k]
            b[k] = b[p]
            b[p] = t
        akk = A[k, k]
        for i in range(k + 1, n):
            f = A[i, k] / akk
            if f != 0.0:
                for j in range(k, n):
                    A[i, j] -= f * A[k, j]
                b[i] -= f * b[k]
    for k in range(n - 1, -1, -1):
        s = b[k]
        for j in range(k + 1, n):
            s -= A[k, j] * b[j]
        b[k] = s / A[k, k]
    return True


@njit(cache=True)
def _bpp_one(x, H, max_iter, feas_tol, Y_out, L_out):  # pragma: no cover - njit
    """Block principal pivoting (active-set Newton) for one LCP sample.

    Returns ``(ok, iterations)``. On success writes the reflected state into
    ``Y_out`` and the boundary push into ``L_out``.
    """
    n = x.shape[0]
    inF = np.empty(n, dtype=np.bool_)
    # Heuristic start: free the coordinates that are currently infeasible (x<0).
    for i in range(n):
        inF[i] = x[i] < 0.0

    L = np.zeros(n)
    Y = np.empty(n)
    idx = np.empty(n, dtype=np.int64)
    A = np.empty((n, n))
    rhs = np.empty(n)

    for it in range(max_iter):
        kb = 0
        for i in range(n):
            if inF[i]:
                idx[kb] = i
                kb += 1
            L[i] = 0.0

        if kb > 0:
            for a in range(kb):
                ia = idx[a]
                rhs[a] = -x[ia]
                for c in range(kb):
                    A[a, c] = H[ia, idx[c]]
            if not _solve_small(A, rhs, kb):
                return False, it
            for a in range(kb):
                L[idx[a]] = rhs[a]

        # Y = x + H L
        for i in range(n):
            yi = x[i]
            for j in range(n):
                yi += H[i, j] * L[j]
            Y[i] = yi

        # Flip infeasible indices: free L<0 -> tight; tight Y<0 -> free.
        nbad = 0
        for i in range(n):
            if inF[i]:
                if L[i] < -feas_tol:
                    inF[i] = False
                    nbad += 1
            else:
                if Y[i] < -feas_tol:
                    inF[i] = True
                    nbad += 1

        if nbad == 0:
            for i in range(n):
                li = L[i]
                yi = Y[i]
                if li < 0.0:
                    li = 0.0
                if yi < 0.0:
                    yi = 0.0
                L_out[i] = li
                Y_out[i] = yi
            return True, it + 1

    return False, max_iter


@njit(cache=True, parallel=True)
def _batched_newton_lemke(X, H, M_scaled, row_scale, col_scale,
                          bpp_max_iter, feas_tol,
                          lemke_max_pivots, lemke_tol, lemke_pivot_tol,
                          lemke_residual_tol):  # pragma: no cover - njit
    B, n = X.shape
    Y = np.empty_like(X)
    L = np.empty_like(X)
    ok = np.zeros(B, dtype=np.bool_)
    used_lemke = np.zeros(B, dtype=np.bool_)
    iters = np.zeros(B, dtype=np.int64)
    for b in prange(B):
        good, nit = _bpp_one(X[b], H, bpp_max_iter, feas_tol, Y[b], L[b])
        if good:
            ok[b] = True
            iters[b] = nit
        else:
            # Escalate this single path to the exact Lemke pivot -- still on the
            # worker thread, never a Python-level serial loop.
            g2, npiv = _solve_one_scaled_lemke(
                X[b], M_scaled, H, row_scale, col_scale,
                lemke_max_pivots, lemke_tol, lemke_pivot_tol, lemke_residual_tol,
                Y[b], L[b])
            ok[b] = g2
            used_lemke[b] = True
            iters[b] = npiv
    return Y, L, ok, used_lemke, iters


class NewtonLemkeReflection:
    """Block-principal-pivoting (active-set Newton) LCP solver, Lemke fallback."""

    def __init__(self, H, bpp_max_iter: int | None = None, feas_tol: float = 1e-10,
                 residual_tol: float = 1e-8, lemke_tol: float = 1e-10,
                 lemke_pivot_tol: float = 1e-12, lemke_max_pivots: int | None = None,
                 scale_iterations: int = 10):
        self.H = np.ascontiguousarray(H, dtype=np.float64)
        self.dim = self.H.shape[0]
        # A P-matrix solution needs at most one flip per index, so O(dim) blocks
        # suffice; allow a few rounds of headroom before escalating to Lemke.
        self.bpp_max_iter = int(bpp_max_iter) if bpp_max_iter else max(20, 4 * self.dim)
        self.feas_tol = float(feas_tol)
        self.residual_tol = float(residual_tol)
        self.lemke_tol = float(lemke_tol)
        self.lemke_pivot_tol = float(lemke_pivot_tol)
        self.lemke_max_pivots = (int(lemke_max_pivots) if lemke_max_pivots is not None
                                 else max(1000, 50 * self.dim))
        # The Lemke fallback reuses the equilibrated matrix from lemke_lcp.
        row_scale, col_scale, M_scaled = equilibrate_lcp_matrix(self.H, n_iter=scale_iterations)
        self.row_scale = np.ascontiguousarray(row_scale, dtype=np.float64)
        self.col_scale = np.ascontiguousarray(col_scale, dtype=np.float64)
        self.M_scaled = np.ascontiguousarray(M_scaled, dtype=np.float64)
        # diagnostics from the last batch
        self.last_ok = None
        self.last_used_lemke = None
        self.last_iters = None

    @property
    def backend(self) -> str:
        return "newton_lemke" if _HAS_NUMBA else "newton_lemke-python"

    def _solve_batch(self, X: np.ndarray):
        Y, L, ok, used_lemke, iters = _batched_newton_lemke(
            X, self.H, self.M_scaled, self.row_scale, self.col_scale,
            self.bpp_max_iter, self.feas_tol,
            self.lemke_max_pivots, self.lemke_tol, self.lemke_pivot_tol,
            self.residual_tol)
        self.last_ok = ok
        self.last_used_lemke = used_lemke
        self.last_iters = iters
        if not np.all(ok):
            failed = np.where(~ok)[0]
            raise RuntimeError(
                f"newton_lemke LCP failed for {failed.size}/{X.shape[0]} samples "
                f"(both BPP and Lemke); first={int(failed[0])}")
        return Y, L

    def solve(self, X: np.ndarray) -> ReflectionResult:
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y, L = self._solve_batch(X)
        res = lcp_residual(X, Y, self.H)
        return ReflectionResult(reflected=Y, boundary_push=L, lcp_residual=res,
                                iterations=int(self.last_iters.max(initial=0)),
                                fallback_count=int(self.last_used_lemke.sum()))

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = X.copy()
        need = np.where(~np.all(X >= 0.0, axis=1))[0]
        if need.size == 0:
            self.last_ok = np.ones(X.shape[0], dtype=bool)
            self.last_used_lemke = np.zeros(X.shape[0], dtype=bool)
            self.last_iters = np.zeros(X.shape[0], dtype=np.int64)
            return Y
        Ys, _Ls = self._solve_batch(np.ascontiguousarray(X[need]))
        Y[need] = Ys
        return Y
