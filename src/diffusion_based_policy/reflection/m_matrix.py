"""M-matrix reflection backend.

This backend uses the same LCP solver as the standard CPU implementation, but
first checks that the reflection matrix is a nonsingular M-matrix.
"""
from __future__ import annotations

import numpy as np

from .pgs_enum_lcp import PGSEnumReflection


class MMatrixReflection(PGSEnumReflection):
    """LCP reflection solver restricted to nonsingular M-matrices."""

    def __init__(self, H, matrix_tol: float = 1e-10, **kwargs):
        H_arr = np.asarray(H, dtype=np.float64)
        off_diag = H_arr - np.diag(np.diag(H_arr))
        inv = np.linalg.inv(H_arr)
        if (
            np.any(np.diag(H_arr) <= 0.0)
            or np.any(off_diag > matrix_tol)
            or np.any(inv < -matrix_tol)
        ):
            raise ValueError("reflection matrix is not a nonsingular M-matrix")
        super().__init__(H_arr, **kwargs)

    @property
    def backend(self) -> str:
        return "m_matrix"
