"""Thread-count guards, applied before numpy / numba / tensorflow load.

numba's parallel reflection kernel and TensorFlow each spawn their own thread
pools. With numba defaulting to one thread per core, the two pools plus the
interpreter can exhaust the per-user / cgroup thread limit and TensorFlow then
aborts with ``pthread_create() failed`` (errno 11). Capping both pools to a
modest default keeps them coexisting in a single process. Real environments
(e.g. a SLURM allocation) can raise the caps by exporting the variables before
importing the package.

Import this module first -- ``import diffusion_based_policy._env  # noqa: F401`` -- so the
variables are set before the C extensions read them.
"""
from __future__ import annotations

import os

_DEFAULTS = {
    "NUMBA_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "TF_NUM_INTEROP_THREADS": "2",
    "TF_NUM_INTRAOP_THREADS": "4",
    # allow a duplicate OpenMP runtime (numba + TF) instead of aborting
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}

for _k, _v in _DEFAULTS.items():
    os.environ.setdefault(_k, _v)
