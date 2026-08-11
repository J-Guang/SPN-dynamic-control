"""Shared bootstrap: make ``src/`` importable when scripts run from publication/.

Import this first in every script:  ``import _bootstrap  # noqa: F401``.
"""
from __future__ import annotations

import os
import sys

# Cap thread pools before numpy / numba / tensorflow load (see diffusion_based_policy._env).
for _k, _v in {
    "NUMBA_NUM_THREADS": "4", "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "TF_NUM_INTEROP_THREADS": "2", "TF_NUM_INTRAOP_THREADS": "4",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}.items():
    os.environ.setdefault(_k, _v)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def publication_root() -> str:
    return _ROOT
