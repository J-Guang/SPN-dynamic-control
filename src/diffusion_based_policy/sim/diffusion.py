"""Reflected reference-diffusion simulation (math_foundation.md Section 4.1).

The reference reflected diffusion is

    dZ = m dt + sigma dW + H dL,    Z in R^I_+,

with sigma sigma^T = Gamma and m the (constant) reference state drift. By
default m = zeta, i.e. the nominal reference rule u = beta (theta_bar = 0); a
config may override m to push paths into a region of interest. The BSDE identity
is exact for any reference drift, so m only controls *which* states are sampled.

Each Euler step integrates the unreflected increment x = z + m dt + sigma dW and
projects with the Skorokhod reflection (H) backend.
"""
from __future__ import annotations

import numpy as np

from ..bcp import BCPModel
from ..reflection import get_solver


def _initial_states(rng, init: str, num_paths: int, dim: int, scale: float) -> np.ndarray:
    init = (init or "zeros").lower()
    if init == "zeros":
        return np.zeros((num_paths, dim))
    if init == "uniform":
        return rng.uniform(0.0, scale, size=(num_paths, dim))
    if init == "exponential":
        return rng.exponential(scale, size=(num_paths, dim))
    raise ValueError(f"unknown init distribution '{init}'")


def simulate_reference_paths(
    model: BCPModel,
    num_paths: int,
    num_steps: int,
    horizon: float,
    seed: int,
    backend: str = "numba",
    reference_drift: np.ndarray | None = None,
    init: str = "zeros",
    init_scale: float = 1.0,
) -> dict:
    """Simulate reflected reference paths.

    Returns a dict with:
      z_path  (num_paths, num_steps + 1, I)  reflected states at each node
      dw      (num_paths, num_steps, I)       diffusion increments sigma dW
      l_incr  (num_paths, num_steps, I)       boundary push increments dL
      dt, horizon, num_steps, reference_drift
    """
    I = model.I
    dt = float(horizon) / num_steps
    sigma = model.sigma
    m = model.zeta.copy() if reference_drift is None else np.asarray(reference_drift, float)
    solver = get_solver(backend, model.H)
    rng = np.random.default_rng(seed)

    z0 = _initial_states(rng, init, num_paths, I, init_scale)
    z_path, dw, l_incr, _ = _run_segment(solver, z0, num_steps, dt, sigma, m, rng)

    return {
        "z_path": z_path,
        "dw": dw,
        "l_incr": l_incr,
        "dt": dt,
        "horizon": float(horizon),
        "num_steps": num_steps,
        "reference_drift": m,
    }


def _run_segment(solver, z0: np.ndarray, num_steps: int, dt: float,
                 sigma: np.ndarray, m: np.ndarray, rng) -> tuple:
    """Integrate ``num_steps`` Euler-reflected steps from z0.

    Returns (z_path (P,N+1,I), dw (P,N,I), l_incr (P,N,I), z_end (P,I)).
    """
    P, I = z0.shape
    sqrt_dt = np.sqrt(dt)
    z = z0
    z_path = np.empty((P, num_steps + 1, I))
    dw = np.empty((P, num_steps, I))
    z_path[:, 0, :] = z
    for k in range(num_steps):
        dw_sigma = (rng.standard_normal((P, I)) * sqrt_dt) @ sigma.T
        z = solver.project(z + m * dt + dw_sigma)   # fast exact reflected state
        dw[:, k, :] = dw_sigma
        z_path[:, k + 1, :] = z
    # The boundary local-time path L (Y = X + H L) is not needed by the chi = 0
    # objective and is the dominant per-step cost, so it is not materialized here.
    l_incr = np.zeros((P, num_steps, I))
    return z_path, dw, l_incr, z


class ReferencePathSampler:
    """Reusable sampler yielding deterministic reference-path batches.

    Batch ``b`` uses seed ``base_seed + b`` so the stream is reproducible and
    independent across batches.
    """

    def __init__(self, model: BCPModel, num_steps: int, horizon: float,
                 backend: str = "numba", reference_drift: np.ndarray | None = None,
                 init: str = "zeros", init_scale: float = 1.0, base_seed: int = 0):
        self.model = model
        self.num_steps = num_steps
        self.horizon = horizon
        self.backend = backend
        self.reference_drift = reference_drift
        self.init = init
        self.init_scale = init_scale
        self.base_seed = base_seed

    def batch(self, num_paths: int, index: int = 0) -> dict:
        return simulate_reference_paths(
            self.model, num_paths=num_paths, num_steps=self.num_steps,
            horizon=self.horizon, seed=self.base_seed + index,
            backend=self.backend, reference_drift=self.reference_drift,
            init=self.init, init_scale=self.init_scale,
        )


class ContinuousReferenceSampler:
    """Stateful sampler that runs ONE long continuous reflected path per row.

    This matches the training scheme of the published runs: the path starts at
    the origin, a ``warmup_segments`` burn-in brings it onto the (reflected)
    stationary distribution, and thereafter each call to ``advance`` returns the
    next ``num_steps``-step segment *continuing from where the previous segment
    ended* (the per-row state carries over). Consecutive BSDE residual segments
    therefore start from stationary-distributed states instead of always from 0,
    so the value/gradient networks see the whole occupied region rather than a
    tiny ball around the origin.

    The RNG is seeded once and advanced deterministically, so a full training run
    is reproducible given the seed.
    """

    def __init__(self, model: BCPModel, num_paths: int, num_steps: int,
                 horizon: float, backend: str = "numba",
                 reference_drift: np.ndarray | None = None,
                 init: str = "zeros", init_scale: float = 1.0,
                 seed: int = 0, warmup_segments: int = 0):
        self.model = model
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.horizon = float(horizon)
        self.dt = self.horizon / num_steps
        self.sigma = model.sigma
        self.m = (model.zeta.copy() if reference_drift is None
                  else np.asarray(reference_drift, float))
        self.solver = get_solver(backend, model.H)
        self.rng = np.random.default_rng(seed)
        self.state = _initial_states(self.rng, init, num_paths, model.I, init_scale)
        self.segments_done = 0
        for _ in range(int(warmup_segments)):
            self._step()  # burn-in toward stationarity; discarded

    def _step(self) -> dict:
        z_path, dw, l_incr, z_end = _run_segment(
            self.solver, self.state, self.num_steps, self.dt,
            self.sigma, self.m, self.rng)
        self.state = z_end                 # carry the path forward (continuity)
        self.segments_done += 1
        return {
            "z_path": z_path, "dw": dw, "l_incr": l_incr, "dt": self.dt,
            "horizon": self.horizon, "num_steps": self.num_steps,
            "reference_drift": self.m,
        }

    def advance(self) -> dict:
        """Return the next continuous segment and move the path forward."""
        return self._step()


def _run_segment_on_policy(solver, z0: np.ndarray, num_steps: int, dt: float,
                           sigma: np.ndarray, model: BCPModel, grad_fn, a: float,
                           rng) -> tuple:
    """Euler-reflected segment whose per-step drift is the behaviour policy:

        g_k   = grad_fn(z_k)                         # frozen behaviour gradient
        mu_k  = zeta + theta(g_k) @ R.T              # on-policy state drift
        z_{k+1} = Skorokhod(z_k + mu_k dt + sigma dW_k)

    Returns (z_path, dw, rdz, l_incr, z_end) where ``rdz`` = mu_k - zeta = the
    per-step reference-drift correction (m - zeta) the BSDE driver subtracts.
    """
    P, I = z0.shape
    sqrt_dt = np.sqrt(dt)
    zeta = model.zeta
    z = z0
    z_path = np.empty((P, num_steps + 1, I))
    dw = np.empty((P, num_steps, I))
    rdz = np.empty((P, num_steps, I))
    z_path[:, 0, :] = z
    for k in range(num_steps):
        mu = model.policy_ref_drift(grad_fn(z), a)        # (P, I) = zeta + theta R^T
        dw_sigma = (rng.standard_normal((P, I)) * sqrt_dt) @ sigma.T
        z = solver.project(z + mu * dt + dw_sigma)
        dw[:, k, :] = dw_sigma
        rdz[:, k, :] = mu - zeta
        z_path[:, k + 1, :] = z
    l_incr = np.zeros((P, num_steps, I))
    return z_path, dw, rdz, l_incr, z


class OnPolicyContinuousSampler:
    """Continuous chained sampler driven by a frozen behaviour policy.

    Same chaining/warmup scheme as ``ContinuousReferenceSampler`` but the drift
    at each step is ``mu(z) = zeta + theta(z) @ R.T`` with ``theta`` the
    behaviour-policy allocation (``model.extract_theta(grad_fn(z), a)``). Each
    segment also carries the per-step reference-drift correction ``rdz`` so the
    loss subtracts the right (state-dependent) ``m - zeta``.
    """

    def __init__(self, model: BCPModel, num_paths: int, num_steps: int,
                 horizon: float, grad_fn, behavior_bound: float,
                 backend: str = "numba", init: str = "zeros",
                 init_scale: float = 1.0, seed: int = 0, warmup_segments: int = 0,
                 chain: bool = False):
        self.model = model
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.horizon = float(horizon)
        self.dt = self.horizon / num_steps
        self.sigma = model.sigma
        self.grad_fn = grad_fn
        self.a = float(behavior_bound)
        self.solver = get_solver(backend, model.H)
        self.rng = np.random.default_rng(seed)
        # chain=False (default): each segment restarts from a fresh ``init`` draw,
        # so paths stay near the origin where the FROZEN behaviour policy is in
        # distribution. Chaining a destabilising/out-of-distribution on-policy
        # drift would let paths wander to states where theta is garbage (and
        # starve the origin), so on-policy reference sampling resets by default.
        self.chain = bool(chain)
        self.init = init
        self.init_scale = init_scale
        self.state = _initial_states(self.rng, init, num_paths, model.I, init_scale)
        self.segments_done = 0
        if self.chain:
            for _ in range(int(warmup_segments)):
                self._step()

    def _step(self) -> dict:
        if not self.chain:
            self.state = _initial_states(self.rng, self.init, self.num_paths,
                                         self.model.I, self.init_scale)
        z_path, dw, rdz, l_incr, z_end = _run_segment_on_policy(
            self.solver, self.state, self.num_steps, self.dt,
            self.sigma, self.model, self.grad_fn, self.a, self.rng)
        self.state = z_end
        self.segments_done += 1
        return {"z_path": z_path, "dw": dw, "rdz": rdz, "l_incr": l_incr,
                "dt": self.dt, "horizon": self.horizon, "num_steps": self.num_steps}

    def advance(self) -> dict:
        return self._step()
