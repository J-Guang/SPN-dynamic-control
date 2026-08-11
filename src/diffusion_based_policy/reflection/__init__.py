"""Reflection backends and factory.

    solver = get_solver("newton_lemke", H)  # active-set Newton + Lemke fallback
    solver = get_solver("pgs_enum", H)       # proj. Gauss-Seidel + basis enumeration
    solver = get_solver("lemke", H)          # scaled Lemke pivoting
    result = solver.solve(X)                 # ReflectionResult(reflected, boundary_push, ...)

All listed backends are correct for any valid reflection matrix H (the
Harrison-Reiman P-matrix condition); they differ in algorithm and speed, not in
the class of H they accept:

  * ``pgs_enum``     -- projected Gauss-Seidel with an exact 2^dim basis-enum
                        fallback (vectorized for dim<=4). Fast for small dim, but
                        PGS does not converge for a non-H-matrix H (e.g. bigstep),
                        which then drop to the serial enumeration -- single core.
                        Aliases: ``numba``, ``cpu``, ``numpy``.
  * ``lemke``        -- scaled batched Lemke pivoting. Aliases: ``scaled_lemke``.
  * ``newton_lemke`` -- block-principal-pivoting (active-set Newton) primary with
                        a Lemke fallback, all inside one numba-parallel kernel;
                        multi-core and polynomial in dim even when H is a
                        non-H-matrix P-matrix. Aliases: ``newton``, ``bpp``.

``m_matrix`` is an *opt-in* stricter variant that additionally requires H to be a
nonsingular M-matrix (holds for Pesic-Williams but NOT crisscross / bigstep); it
is excluded from ``available_backends`` and must be requested explicitly. ``gpu``
is a TensorFlow basis-enumeration backend (requires a GPU node).
"""
from __future__ import annotations

import numpy as np

from .base import ReflectionResult, ReflectionSolver, lcp_residual
from .lemke_lcp import LemkeLCPReflection
from .m_matrix import MMatrixReflection
from .newton_lemke import NewtonLemkeReflection
from .pgs_enum_lcp import PGSEnumReflection

__all__ = [
    "ReflectionResult",
    "ReflectionSolver",
    "lcp_residual",
    "LemkeLCPReflection",
    "MMatrixReflection",
    "NewtonLemkeReflection",
    "PGSEnumReflection",
    "get_solver",
    "available_backends",
]


def _tf_importable() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


def available_backends() -> list[str]:
    """Generally-applicable backends (valid for any P-matrix H).

    ``newton_lemke`` is the canonical default for the experiments. ``m_matrix``
    is intentionally excluded: it is opt-in only (see module docstring) because
    it rejects non-M-matrix reflection matrices such as crisscross / bigstep.
    """
    backends = ["pgs_enum", "lemke", "newton_lemke"]
    if _tf_importable():
        backends.append("gpu")
    return backends


def get_solver(backend: str, H: np.ndarray, **kwargs) -> ReflectionSolver:
    backend = backend.lower()
    if backend in ("pgs_enum", "pgs-enum", "numba", "cpu", "numpy"):
        return PGSEnumReflection(H, **kwargs)
    if backend in ("lemke", "scaled_lemke", "scaled-lemke"):
        return LemkeLCPReflection(H, **kwargs)
    if backend in ("newton_lemke", "newton", "bpp", "newton-lemke"):
        return NewtonLemkeReflection(H, **kwargs)
    if backend in ("m_matrix", "m-matrix", "mmatrix"):
        return MMatrixReflection(H, **kwargs)
    if backend in ("gpu", "tf", "tensorflow"):
        from .gpu_lcp import GPUBasisReflection
        return GPUBasisReflection(H, **kwargs)
    raise ValueError(f"unknown reflection backend '{backend}'")
