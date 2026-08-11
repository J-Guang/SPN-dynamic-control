"""Prelimit queueing-network simulator (math_foundation.md Sections 1, 5).

Vectorized next-event (CTMC) simulation of the scale-n network under a lifted
BCP policy supplied as a gradient function g(z). Tracks the discounted objective

    J^{(n)}(q) = E[ int_0^inf e^{-rho t} h.Q(t) dt + sum rejected-job charges ],

with the prelimit per-rejected-job cost c^{D,(n)}_k = spec.rejection_cost_k. Diffusion horizon T
maps to original time n T (the heavy-traffic time speed-up), so e^{-rho (nT)} =
e^{-gamma T}. Returns cost decomposition, station utilization, and the
n^{-3/2}-scaled Brownian-comparable value.

Events per jump (state q held constant between events):
  * processing resource k: the policy-selected activity completes at rate mu_j;
    on completion q += P_.j.
  * input resource k: arrival opportunity at rate mu_bar_k; on firing, admit/
    route (q += P_.j) or reject (charge e^{-rho t} c^{D,(n)}_k).
"""
from __future__ import annotations

import numpy as np

from ..bcp import BCPModel
from ..policies.vectorized import VectorizedPolicy


def _input_opportunity_prelimit(model: BCPModel) -> np.ndarray:
    """mu_bar_k from prelimit rates (arrival rate of each input stream)."""
    spec = model.spec
    mb = np.zeros(spec.num_resources)
    for k in spec.input_resources:
        block = spec.resource_block(k)
        mb[k] = float(np.max(spec.rates[block])) if block.size else 0.0
    return mb


def simulate_prelimit(
    model: BCPModel,
    grad_fn,
    num_paths: int,
    horizon: float,
    seed: int,
    max_jumps: int | None = None,
    q0: np.ndarray | None = None,
    work_conserving: bool = False,
) -> dict:
    spec = model.spec
    n = model.n
    rho = spec.discount
    sqrt_n = np.sqrt(n)
    P = spec.buffer_change                       # (I, J)
    mu = spec.rates                              # (J,)
    h = spec.holding_cost                        # (I,)
    I, J = P.shape
    K = spec.num_resources

    policy = VectorizedPolicy(model, work_conserving=work_conserving)
    mu_bar = _input_opportunity_prelimit(model)  # (K,)
    cD_n = spec.rejection_cost                   # (K,) prelimit per-job charge c^{D,(n)}
    is_input = spec.is_input
    proc_res = spec.processing_resources
    input_res = spec.input_resources

    T_orig = n * horizon
    max_jumps = max_jumps or int(np.ceil(T_orig * (mu_bar.sum() + mu.max() * len(proc_res)) * 2 + 1000))

    rng = np.random.default_rng(seed)
    M = num_paths
    q = np.zeros((M, I)) if q0 is None else np.broadcast_to(
        np.asarray(q0, float), (M, I)).copy()
    t = np.zeros(M)
    holding = np.zeros(M)
    rejection = np.zeros(M)
    busy_time = np.zeros((M, len(proc_res)))
    feas_idle_time = np.zeros((M, len(proc_res)))   # idle while a queue is servable
    total_time = np.zeros(M)
    arrivals = np.zeros(M)
    rejects = np.zeros(M)
    # per-station consumption blocks (for the "could have served" check) and
    # per-input-stream arrival/rejection counters
    proc_blocks_C = [spec.consumption[:, spec.resource_block(k)] for k in proc_res]
    n_in = len(input_res)
    arr_by = np.zeros(n_in)
    rej_by = np.zeros(n_in)

    jumps_done = 0
    for _ in range(max_jumps):
        active = t < T_orig
        if not active.any():
            break
        jumps_done += 1
        z = q / sqrt_n
        g = grad_fn(z)
        g = np.atleast_2d(np.asarray(g, dtype=np.float64))
        selected = policy.select(q, g)            # (M, K)

        rates = np.zeros((M, K))
        for k in proc_res:
            sel = selected[:, k]
            valid = sel >= 0
            rates[valid, k] = mu[sel[valid]]
        for k in input_res:
            rates[:, k] = mu_bar[k]
        Lam = rates.sum(axis=1)
        Lam_safe = np.where(Lam > 0, Lam, 1.0)

        tau = rng.exponential(1.0 / Lam_safe)
        tau = np.where(active, tau, 0.0)
        t_old = t
        t_new = t + tau
        # The event at t_new only occurs inside the finite horizon. If the next
        # exponential jump overshoots T_orig, the state is held constant on
        # [t, T_orig] with no event: integrate holding/utilization only up to the
        # horizon and apply no service/admission/rejection event.
        within = active & (t_new <= T_orig)
        tau_eff = np.where(active, np.minimum(tau, np.maximum(T_orig - t, 0.0)), 0.0)
        t_cap = t_old + tau_eff                        # = min(t_new, T_orig)

        # discounted holding cost over [t_old, min(t_new, T_orig)] at constant q
        disc = (np.exp(-rho * t_old) - np.exp(-rho * t_cap)) / rho
        hq = q @ h
        holding += np.where(active, hq * disc, 0.0)

        # utilization (busy/idle time only within the horizon); idle-while-servable
        # separates a true work shortage from a policy choice to idle (index <= 0).
        for kk, k in enumerate(proc_res):
            busy = selected[:, k] >= 0
            busy_time[:, kk] += np.where(active & busy, tau_eff, 0.0)
            Cb = proc_blocks_C[kk]                          # (I, |block|)
            servable = np.all(q[:, :, None] >= Cb[None], axis=1).any(axis=1)
            feas_idle = (~busy) & servable
            feas_idle_time[:, kk] += np.where(active & feas_idle, tau_eff, 0.0)
        total_time += np.where(active, tau_eff, 0.0)

        # pick which resource fires
        u = rng.random(M) * Lam_safe
        cum = np.cumsum(rates, axis=1)
        fired = (cum < u[:, None]).sum(axis=1)
        fired = np.clip(fired, 0, K - 1)

        sel_fired = selected[np.arange(M), fired]      # chosen activity (or -1)
        admit = sel_fired >= 0
        idx_safe = np.where(admit, sel_fired, 0)
        delta = P[:, idx_safe].T                        # (M, I)
        apply = within & admit                          # apply event only within horizon
        q = q + np.where(apply[:, None], delta, 0.0)
        q = np.maximum(q, 0.0)

        input_fired = is_input[fired]
        arrivals += np.where(within & input_fired, 1.0, 0.0)
        reject = input_fired & (~admit)
        rejection += np.where(within & reject, np.exp(-rho * t_new) * cD_n[fired], 0.0)
        rejects += np.where(within & reject, 1.0, 0.0)
        for ii, k in enumerate(input_res):           # per-stream arrival/reject counts
            fk = within & (fired == k)
            arr_by[ii] += float(fk.sum())
            rej_by[ii] += float((fk & (~admit)).sum())

        # advance: to t_new if the event occurred, else to the horizon (path stops)
        t = np.where(active, np.where(within, t_new, T_orig), t)

    cost = holding + rejection
    n32 = n ** 1.5
    tt = max(total_time.sum(), 1e-12)
    util = (busy_time.sum(axis=0) / tt)
    idle_feas = (feas_idle_time.sum(axis=0) / tt)
    rej_rate = np.divide(rej_by, arr_by, out=np.zeros_like(rej_by), where=arr_by > 0)
    return {
        "cost_mean": float(cost.mean()),
        "cost_stderr": float(cost.std(ddof=1) / np.sqrt(M)) if M > 1 else 0.0,
        "cost_scaled_mean": float(cost.mean() / n32),
        "holding": float(holding.mean()),
        "rejection": float(rejection.mean()),
        "utilization": util.tolist(),
        "utilization_labels": [spec.resource_labels[k] for k in proc_res],
        "idle_when_servable": idle_feas.tolist(),
        "reject_rate_by_stream": rej_rate.tolist(),
        "reject_stream_labels": [spec.resource_labels[k] for k in input_res],
        "reject_fraction": float(rejects.sum() / max(arrivals.sum(), 1.0)),
        "jumps": jumps_done,
        "num_paths": M,
        "horizon": horizon,
        "T_orig": T_orig,
        # mean simulated (original-time) horizon per path; with the horizon cap
        # this equals T_orig once paths reach the horizon (used to verify no
        # post-horizon overshoot).
        "mean_sim_time": float(total_time.mean()),
        "max_sim_time": float(t.max()),
    }
