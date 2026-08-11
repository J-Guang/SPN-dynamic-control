"""``pgs_enum`` reflection backend: projected Gauss-Seidel + basis enumeration.

Solves  Y = X + H L,  Y,L >= 0,  Y^T L = 0.  For ``dim <= _VECTOR_ENUM_MAX_DIM``
the exact 2^dim basis enumeration runs as a numba-parallel kernel (the sampling
hot path); for larger dim it uses projected Gauss-Seidel with an exact basis-enum
fallback. If numba is unavailable the same algorithm runs in pure numpy.

(Formerly the ``numba`` backend -- a misnomer, since every CPU backend uses
numba; named after its algorithm now. ``numba`` / ``cpu`` / ``numpy`` remain
accepted aliases in ``get_solver``.)
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


_VECTOR_ENUM_MAX_DIM = 8


@njit(cache=True, fastmath=True, parallel=True)
def _batched_pgs(X, M, Mdiag, max_sweeps, tol):  # pragma: no cover - njit
    """Per-sample projected Gauss-Seidel parallelized across the batch.

    Returns (W, Yl, ok) with W = reflected state, Yl = local time, ok = LCP check.
    """
    B, n = X.shape
    Yl = np.zeros((B, n))
    W = X.copy()
    ok = np.zeros(B, dtype=np.bool_)
    check = 1e-8
    for b in prange(B):
        for _sweep in range(max_sweeps):
            max_delta = 0.0
            for i in range(n):
                Mii = Mdiag[i]
                qtilde = W[b, i] - Mii * Yl[b, i]
                y_new = 0.0 if qtilde >= 0.0 else -qtilde / Mii
                delta = y_new - Yl[b, i]
                if delta != 0.0:
                    Yl[b, i] = y_new
                    for j in range(n):
                        W[b, j] += delta * M[j, i]
                    ad = delta if delta >= 0.0 else -delta
                    if ad > max_delta:
                        max_delta = ad
            if max_delta < tol:
                break
        for i in range(n):
            if Yl[b, i] < 0.0:
                Yl[b, i] = 0.0
        for i in range(n):
            s = X[b, i]
            for j in range(n):
                s += M[i, j] * Yl[b, j]
            W[b, i] = s
        good = True
        for i in range(n):
            if Yl[b, i] < -check or W[b, i] < -check:
                good = False
                break
            if Yl[b, i] > check and W[b, i] > check:
                good = False
                break
        ok[b] = good
    return W, Yl, ok


@njit(cache=True, fastmath=True, parallel=True)
def _batched_enum_project(X, B_mask, A, HA, valid, tol):  # pragma: no cover - njit
    """Exact basis enumeration parallelized across samples.

    This is the sampling hot path for dim <= _VECTOR_ENUM_MAX_DIM. It avoids the
    large temporary arrays used by the numpy vectorized fallback and avoids the
    Python per-sample fallback that made boundary-heavy bigstep batches serial.
    """
    B, n = X.shape
    num_masks = valid.shape[0]
    Y = np.empty_like(X)
    ok = np.zeros(B, dtype=np.bool_)
    for b in prange(B):
        heuristic = 0
        for i in range(n):
            if X[b, i] < 0.0:
                heuristic |= (1 << i)

        found = False
        for pos in range(num_masks + 1):
            mask = heuristic if pos == 0 else pos - 1
            if pos != 0 and mask == heuristic:
                continue
            if not valid[mask]:
                continue

            feasible = True
            for i in range(n):
                if B_mask[mask, i] > 0.5:
                    l_i = 0.0
                    for j in range(n):
                        l_i -= A[mask, i, j] * X[b, j]
                    if l_i < -tol:
                        feasible = False
                        break
            if not feasible:
                continue

            for i in range(n):
                if B_mask[mask, i] <= 0.5:
                    w_i = X[b, i]
                    for j in range(n):
                        w_i -= HA[mask, i, j] * X[b, j]
                    if w_i < -tol:
                        feasible = False
                        break
            if not feasible:
                continue

            for i in range(n):
                if B_mask[mask, i] > 0.5:
                    Y[b, i] = 0.0
                else:
                    w_i = X[b, i]
                    for j in range(n):
                        w_i -= HA[mask, i, j] * X[b, j]
                    Y[b, i] = w_i if w_i > 0.0 else 0.0
            ok[b] = True
            found = True
            break

        if not found:
            for i in range(n):
                Y[b, i] = X[b, i] if X[b, i] > 0.0 else 0.0
    return Y, ok


class PGSEnumReflection:
    """Reflection solver using the PGS + basis-enumeration LCP algorithm."""

    def __init__(self, H, tol: float = 1e-10, gs_max_sweeps: int = 1000,
                 gs_tol: float = 1e-12, parallel: bool = True,
                 parallel_min_batch: int = 64):
        self.H = np.ascontiguousarray(H, dtype=np.float64)
        self.dim = self.H.shape[0]
        self.tol = tol
        self.gs_max_sweeps = gs_max_sweeps
        self.gs_tol = gs_tol
        self._Mdiag = np.ascontiguousarray(np.diag(self.H))
        self.parallel = bool(parallel) and _HAS_NUMBA
        self.parallel_min_batch = parallel_min_batch
        self._precompute_bases()
        # Vectorized exact basis-enum tensors for low-dimensional networks.
        # PGS is fast on interior-heavy batches, but the training sampler (drift
        # toward 0) produces boundary-heavy states where near-degenerate H can
        # fail the strict LCP gate. A Python per-sample enumeration fallback then
        # serializes the sampling hot path. Up through dim 8 there are at most
        # 256 bases, so exact vectorized enumeration is both small and robust.
        self._enum_vec = None
        if self.dim <= _VECTOR_ENUM_MAX_DIM:
            from .gpu_lcp import _precompute_basis_tensors
            B_mask, A, HA, valid = _precompute_basis_tensors(self.H, self.dim)
            self._enum_vec = (B_mask, A, HA, valid)

    @property
    def backend(self) -> str:
        return "pgs_enum" if _HAS_NUMBA else "pgs_enum-numpy"

    def _project_enum_vectorized(self, X: np.ndarray) -> np.ndarray:
        """Exact reflected state Y via vectorized basis enumeration (no PGS)."""
        B_mask, A, HA, valid = self._enum_vec
        tol = self.tol
        L_all = -np.einsum("mij,bj->mbi", A, X)
        W_all = X[None] - np.einsum("mij,bj->mbi", HA, X)
        Bm = B_mask[:, None, :]
        Nm = 1.0 - Bm
        feas = ((L_all * Bm + Nm).min(-1) >= -tol) & \
               ((W_all * Nm + Bm).min(-1) >= -tol) & valid[:, None]
        first = np.argmax(feas, axis=0)
        rows = np.arange(X.shape[0])
        return np.maximum(W_all[first, rows] * (1.0 - B_mask[first]), 0.0)

    def _project_enum(self, X: np.ndarray) -> np.ndarray:
        """Exact reflected state Y for precomputed low-dimensional bases."""
        if _HAS_NUMBA and self.parallel and X.shape[0] >= self.parallel_min_batch:
            B_mask, A, HA, valid = self._enum_vec
            Y, ok = _batched_enum_project(X, B_mask, A, HA, valid, self.tol)
            if np.all(ok):
                return Y
            for local in np.where(~ok)[0]:
                Yb, _ = self._enum_solve(X[local])
                Y[local] = Yb
            return Y
        return self._project_enum_vectorized(X)

    # --- exact basis enumeration (fallback) --------------------------------
    def _precompute_bases(self) -> None:
        dim, H = self.dim, self.H
        self._bases = [None] * (1 << dim)
        for mask in range(1 << dim):
            B_idx = [i for i in range(dim) if mask & (1 << i)]
            N_idx = [i for i in range(dim) if not (mask & (1 << i))]
            if not B_idx:
                self._bases[mask] = (np.array(B_idx, np.intp), np.array(N_idx, np.intp),
                                     None, None)
                continue
            B = np.array(B_idx, np.intp)
            N = np.array(N_idx, np.intp)
            H_BB = H[np.ix_(B, B)]
            try:
                inv = np.linalg.inv(H_BB)
            except np.linalg.LinAlgError:
                self._bases[mask] = (B, N, None, None)
                continue
            H_NB = H[np.ix_(N, B)] if N.size else np.empty((0, B.size))
            self._bases[mask] = (B, N, inv, H_NB)

    def _enum_solve(self, x: np.ndarray):
        tol, dim = self.tol, self.dim
        heuristic = 0
        for i in range(dim):
            if x[i] < 0:
                heuristic |= (1 << i)
        order = [heuristic] + [m for m in range(1 << dim) if m != heuristic]
        for mask in order:
            B, N, inv, H_NB = self._bases[mask]
            if B.size == 0:
                if np.all(x >= -tol):
                    return np.maximum(x, 0.0), np.zeros(dim)
                continue
            if inv is None:
                continue
            L_B = inv @ (-x[B])
            if np.any(L_B < -tol):
                continue
            Y = np.zeros(dim)
            if N.size:
                Y_N = x[N] + H_NB @ L_B
                if np.any(Y_N < -tol):
                    continue
                Y[N] = np.maximum(Y_N, 0.0)
            L = np.zeros(dim)
            L[B] = np.maximum(L_B, 0.0)
            return Y, L
        raise RuntimeError(f"no feasible LCP basis for x={x}")

    def _pgs_numpy(self, X):
        """Pure-numpy PGS used when numba is unavailable."""
        B, n = X.shape
        Yl = np.zeros((B, n))
        W = X.copy()
        ok = np.zeros(B, dtype=bool)
        for b in range(B):
            y = np.zeros(n)
            w = X[b].copy()
            for _ in range(self.gs_max_sweeps):
                max_delta = 0.0
                for i in range(n):
                    Mii = self.H[i, i]
                    qtilde = w[i] - Mii * y[i]
                    y_new = 0.0 if qtilde >= 0 else -qtilde / Mii
                    delta = y_new - y[i]
                    if delta != 0.0:
                        y[i] = y_new
                        w += delta * self.H[:, i]
                        max_delta = max(max_delta, abs(delta))
                if max_delta < self.gs_tol:
                    break
            y = np.maximum(y, 0.0)
            w = X[b] + self.H @ y
            Yl[b] = y
            W[b] = w
            ok[b] = not (np.any(y < -1e-8) or np.any(w < -1e-8)
                         or np.any((y > 1e-8) & (w > 1e-8)))
        return W, Yl, ok

    # --- public API --------------------------------------------------------
    def solve(self, X: np.ndarray, tighten: bool = True,
              tighten_tol: float = 1e-9) -> ReflectionResult:
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = X.copy()
        L = np.zeros_like(X)
        all_pos = np.all(X >= 0.0, axis=1)
        need = np.where(~all_pos)[0]
        fallback = 0
        if need.size:
            Xs = np.ascontiguousarray(X[need])
            if _HAS_NUMBA and self.parallel and need.size >= self.parallel_min_batch:
                Ws, Yls, ok = _batched_pgs(Xs, self.H, self._Mdiag,
                                           self.gs_max_sweeps, self.gs_tol)
            else:
                Ws, Yls, ok = self._pgs_numpy(Xs)
            Y[need] = Ws
            L[need] = Yls
            # PGS may leave a few boundary samples short of the LCP tolerance
            # while still passing its loose internal gate. Flag any sample whose
            # nonnegativity / complementarity residual is non-trivial and solve
            # it exactly by basis enumeration. This makes the backend strictly
            # exact rather than tolerance-limited.
            bad = ~ok
            if tighten:
                comp = np.max(np.abs(Ws * Yls), axis=1)
                negY = np.max(np.maximum(-Ws, 0.0), axis=1)
                negL = np.max(np.maximum(-Yls, 0.0), axis=1)
                bad = bad | (comp > tighten_tol) | (negY > tighten_tol) | (negL > tighten_tol)
            for local in np.where(bad)[0]:
                gi = need[local]
                Yb, Lb = self._enum_solve(X[gi])
                Y[gi] = Yb
                L[gi] = Lb
                fallback += 1
        res = lcp_residual(X, Y, self.H)
        return ReflectionResult(reflected=Y, boundary_push=L,
                                lcp_residual=res, iterations=0,
                                fallback_count=fallback)

    def project(self, X: np.ndarray) -> np.ndarray:
        """Fast reflected state Y only -- the sampling hot-path entry point.

        Low dim: exact vectorized basis enumeration (robustly fast on the
        boundary-heavy states the sampler produces, where degenerate-H PGS can
        fail and fall back serially). Larger dim: lean PGS + exact basis-enum
        fallback. Either way this skips the diagnostic ``lcp_residual`` and
        tightening pass that make the full ``solve`` much slower.
        """
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = X.copy()
        need = np.where(~np.all(X >= 0.0, axis=1))[0]
        if need.size == 0:
            return Y
        if self._enum_vec is not None:
            Y[need] = self._project_enum(X[need])
            return Y
        Xs = np.ascontiguousarray(X[need])
        if _HAS_NUMBA and self.parallel and need.size >= self.parallel_min_batch:
            Ws, _Yls, ok = _batched_pgs(Xs, self.H, self._Mdiag,
                                        self.gs_max_sweeps, self.gs_tol)
        else:
            Ws, _Yls, ok = self._pgs_numpy(Xs)
        Y[need] = Ws
        for local in np.where(~ok)[0]:
            gi = need[local]
            Yb, _ = self._enum_solve(X[gi])
            Y[gi] = Yb
        return Y
