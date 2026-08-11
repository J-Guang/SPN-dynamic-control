"""Reflection solver interface.

All backends solve the linear complementarity problem (Skorokhod reflection)

    Y = X + H L,    Y >= 0,    L >= 0,    Y^T L = 0,

where X is the unreflected state, H is the reflection matrix (= R Q from
math_foundation.md Section 3.1), Y is the reflected state, and L is the boundary
local time (the cumulative push). Both numba and GPU backends return the same
ReflectionResult structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class ReflectionResult:
    reflected: np.ndarray          # Y  (N, dim)
    boundary_push: np.ndarray      # L  (N, dim)
    lcp_residual: float            # max LCP violation over the batch
    iterations: int = 0            # solver iterations / fallbacks used
    fallback_count: int = 0        # samples that needed the exact enumeration

    def check(self, tol: float = 1e-6) -> bool:
        return self.lcp_residual <= tol


def lcp_residual(X: np.ndarray, Y: np.ndarray, H: np.ndarray) -> float:
    """Max violation of (Y >= 0, L >= 0, Y.L = 0) given Y and X (L = H^{-1}(Y-X))."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    L = np.linalg.solve(H, (Y - X).T).T
    neg_Y = np.maximum(-Y, 0.0)
    neg_L = np.maximum(-L, 0.0)
    comp = np.abs(Y * L)
    return float(np.max([neg_Y.max(initial=0.0),
                         neg_L.max(initial=0.0),
                         comp.max(initial=0.0)]))


@runtime_checkable
class ReflectionSolver(Protocol):
    """Protocol implemented by every reflection backend."""

    H: np.ndarray

    def solve(self, X: np.ndarray) -> ReflectionResult:
        """Reflect a batch X (N, dim) -> ReflectionResult."""
        ...

    def project(self, X: np.ndarray) -> np.ndarray:
        """Fast path: return only the reflected state Y (N, dim)."""
        ...
