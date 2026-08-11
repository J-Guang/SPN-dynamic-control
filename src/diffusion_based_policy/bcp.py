"""Heavy-traffic data and the modified Brownian Control Problem.

Implements math_foundation.md Sections 2 and 3:
  * Brownian drift          zeta  = P diag(mu_hat) beta
  * control-direction       R     = -P diag(mu*)
  * covariance              Gamma = P diag(mu* o beta) P^T
  * monotonicity matrix     K     (critical processing rows + nonbasic rows)
  * boundary correction     Q     (documented equal-sharing construction, Sec 3.6)
  * reflection matrix       H     = R Q
  * Brownian costs          htilde, c_tilde (Sec 2.3)
  * policy index            pi_j(z; g) = b (g.R_.j + (K^T c_tilde)_j)

Everything is derived from the NetworkSpec plus a small BCPParams block; nothing
is hard-coded per network.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .network import NetworkSpec


@dataclass(frozen=True)
class BCPParams:
    """Section 2/3 heavy-traffic data attached to a network."""

    critical_rates: np.ndarray       # mu*  (J)
    rate_perturbation: np.ndarray    # mu_hat (J)  -- DERIVED: sqrt(n)(mu^(n) - mu*)
    nominal_allocation: np.ndarray   # beta (J)
    nonbasic_activities: tuple[int, ...]  # indices j with beta_j = 0 appended to K
    critical_resources: tuple[int, ...]   # bottleneck processing rows (A beta = 1)
    n: int = 400
    gamma: float = 4.0               # DERIVED: n * rho^(n)  (rho = spec.discount)
    b: float = 20.0
    omega: float = 0.99


def bcp_params_from_dict(d: dict, spec: NetworkSpec) -> BCPParams:
    mu_star = np.asarray(d["critical_rates"], dtype=np.float64)
    beta = np.asarray(d["nominal_allocation"], dtype=np.float64)
    n = int(d.get("n", 400))
    # Everything scale- or beta-determined is DERIVED, not a config input:
    #   mu_hat   = sqrt(n) (mu^(n) - mu*)                       (2.1)
    #   gamma    = n rho^(n)                                    (2.2)
    #   nonbasic = { j : beta_j = 0 }                           (3.1)
    #   critical = { processing k : (A beta)_k = 1 }            (2.1)
    mu_hat = np.sqrt(n) * (spec.rates - mu_star)
    gamma = float(n) * float(spec.discount)
    nonbasic = tuple(int(j) for j in np.where(np.abs(beta) < 1e-12)[0])
    load = spec.resource_use @ beta
    crit = tuple(int(k) for k in range(spec.num_resources)
                 if spec.is_processing[k] and abs(load[k] - 1.0) < 1e-9)
    return BCPParams(
        critical_rates=mu_star,
        rate_perturbation=mu_hat,
        nominal_allocation=beta,
        nonbasic_activities=nonbasic,
        critical_resources=crit,
        n=n,
        gamma=gamma,
        b=float(d.get("b", 20.0)),
        omega=float(d.get("omega", 0.99)),
    )


def rescale_params(params: BCPParams, spec: NetworkSpec, *, n=None, b=None) -> BCPParams:
    """Rebuild params for an overridden scale n or control bound b, re-deriving the
    scale-dependent mu_hat = sqrt(n)(mu^(n)-mu*) and gamma = n rho^(n)."""
    new_n = int(n) if n is not None else params.n
    new_b = float(b) if b is not None else params.b
    return BCPParams(
        critical_rates=params.critical_rates,
        rate_perturbation=np.sqrt(new_n) * (spec.rates - params.critical_rates),
        nominal_allocation=params.nominal_allocation,
        nonbasic_activities=params.nonbasic_activities,
        critical_resources=params.critical_resources,
        n=new_n,
        gamma=float(new_n) * float(spec.discount),
        b=new_b,
        omega=params.omega,
    )


class BCPModel:
    """Bundles a NetworkSpec with its BCPParams and exposes every BCP object."""

    def __init__(self, spec: NetworkSpec, params: BCPParams):
        self.spec = spec
        self.params = params

    # convenience scalars ---------------------------------------------------
    @property
    def I(self) -> int:
        return self.spec.num_buffers

    @property
    def J(self) -> int:
        return self.spec.num_activities

    @property
    def b(self) -> float:
        return self.params.b

    @property
    def gamma(self) -> float:
        return self.params.gamma

    @property
    def n(self) -> int:
        return self.params.n

    # --- Section 2 initial-BCP data ---------------------------------------
    @cached_property
    def zeta(self) -> np.ndarray:
        """zeta = P diag(mu_hat) beta."""
        P = self.spec.buffer_change
        return P @ (self.params.rate_perturbation * self.params.nominal_allocation)

    @cached_property
    def R(self) -> np.ndarray:
        """R = -P diag(mu*)."""
        return -self.spec.buffer_change * self.params.critical_rates[None, :]

    @cached_property
    def Gamma(self) -> np.ndarray:
        """Gamma = P diag(mu* o beta) P^T."""
        P = self.spec.buffer_change
        w = self.params.critical_rates * self.params.nominal_allocation
        return P @ np.diag(w) @ P.T

    @cached_property
    def sigma(self) -> np.ndarray:
        """A square root sigma with sigma sigma^T = Gamma (symmetric PSD sqrt)."""
        G = 0.5 * (self.Gamma + self.Gamma.T)
        vals, vecs = np.linalg.eigh(G)
        vals = np.clip(vals, 0.0, None)
        return (vecs * np.sqrt(vals)) @ vecs.T

    # --- monotonicity matrix K and its cost vector -------------------------
    @cached_property
    def _K_rows(self) -> list[tuple[str, int]]:
        """Ordered description of each K row: ('resource', k) or ('nonbasic', j)."""
        rows: list[tuple[str, int]] = []
        crit = set(self.params.critical_resources)
        for k in range(self.spec.num_resources):
            admissible_proc = self.spec.is_processing[k] and (k in crit)
            admissible_input = self.spec.is_input[k] and bool(self.spec.idle_allowed[k])
            if admissible_proc or admissible_input:
                rows.append(("resource", k))
        for j in self.params.nonbasic_activities:
            rows.append(("nonbasic", int(j)))
        return rows

    @cached_property
    def K_matrix(self) -> np.ndarray:
        """Monotonicity matrix K (rows x J). U = K Y."""
        A = self.spec.resource_use
        rows = []
        for kind, idx in self._K_rows:
            if kind == "resource":
                rows.append(A[idx].copy())
            else:  # nonbasic: append -e_j^T
                e = np.zeros(self.J)
                e[idx] = -1.0
                rows.append(e)
        return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, self.J))

    @cached_property
    def input_opportunity_rate(self) -> np.ndarray:
        """mu_bar_k for input resources: critical rate of the input activities."""
        mu_star = self.params.critical_rates
        mb = np.zeros(self.spec.num_resources)
        for k in self.spec.input_resources:
            block = self.spec.resource_block(k)
            mb[k] = float(np.max(mu_star[block])) if block.size else 0.0
        return mb

    @cached_property
    def c_tilde(self) -> np.ndarray:
        """Brownian resource-control cost, conformable with U = K Y (Sec 2.3).

        Config holds the prelimit per-job costs c^{I,(n)}, c^{D,(n)}; the Brownian
        limit divides by n: bar c^I = c^{I,(n)}/n, bar c^D = c^{D,(n)}/n.

        processing row k:  bar c^I_k = input_cost_k / n               (here zero)
        input row k:       mu_bar_k bar c^D_k = mu_bar_k rejection_cost_k / n
        nonbasic row:      0
        """
        out = []
        mu_bar = self.input_opportunity_rate
        n = self.params.n
        for kind, idx in self._K_rows:
            if kind == "nonbasic":
                out.append(0.0)
            elif self.spec.is_processing[idx]:
                out.append(float(self.spec.input_cost[idx] / n))
            else:  # input
                out.append(float(mu_bar[idx] * self.spec.rejection_cost[idx] / n))
        return np.asarray(out, dtype=np.float64)

    @cached_property
    def h_tilde(self) -> np.ndarray:
        return self.spec.holding_cost.copy()

    # --- boundary matrix Q and reflection H = R Q --------------------------
    @cached_property
    def Q(self) -> np.ndarray:
        """Boundary-correction matrix Q (J x I), built by the equal-sharing
        construction of math_foundation.md Section 3.6."""
        spec = self.spec
        beta = self.params.nominal_allocation
        omega = self.params.omega
        C = spec.consumption
        A = spec.resource_use
        J, I = spec.num_activities, spec.num_buffers
        Q = np.zeros((J, I))
        for i in range(I):
            S_i = np.where(C[i] > 0)[0]                 # activities consuming buffer i
            consumes_any = np.where(C.sum(axis=0) > 0)[0]
            for j in S_i:
                Q[j, i] = beta[j]
            for k in range(spec.num_resources):
                if spec.is_input[k]:
                    continue  # input rows stay zero (no boundary reflection of input)
                r_ki = float(np.sum(A[k, S_i] * beta[S_i]))
                if r_ki == 0.0:
                    continue
                # M_ki = service activities on k not consuming buffer i
                M_ki = [
                    l for l in range(J)
                    if A[k, l] > 0 and C[i, l] == 0 and l in consumes_any
                ]
                if not M_ki:
                    continue
                share = omega * r_ki / len(M_ki)
                for l in M_ki:
                    Q[l, i] -= share
        return Q

    @cached_property
    def H(self) -> np.ndarray:
        """Reflection matrix H = R Q (I x I)."""
        return self.R @ self.Q

    @cached_property
    def chi(self) -> np.ndarray:
        """Boundary cost chi = Q^T K^T c_tilde."""
        return self.Q.T @ (self.K_matrix.T @ self.c_tilde)

    # --- policy index ------------------------------------------------------
    @cached_property
    def index_offset(self) -> np.ndarray:
        """The g-independent part of pi_j / b: (K^T c_tilde)_j."""
        return self.K_matrix.T @ self.c_tilde

    def policy_index(self, g: np.ndarray) -> np.ndarray:
        """pi_j(z; g) = b (g . R_.j + (K^T c_tilde)_j).

        g may be (I,) or (..., I); returns matching (..., J).
        """
        g = np.asarray(g, dtype=np.float64)
        gR = g @ self.R                      # (..., J)
        return self.params.b * (gR + self.index_offset)

    def extract_theta(self, g: np.ndarray, a: float) -> np.ndarray:
        """Behaviour-policy allocation deviation theta (B, J) from gradient g (B, I).

        Generic block-argmax LP (the on-policy reference drift uses
        ``mu = zeta + theta @ R.T``). For each resource block, every activity sits
        at its nominal share ``beta_j * a``; the block's best activity
        (argmax of ``alpha = g R + offset``) additionally takes the whole block
        capacity ``c = sum_j beta_j`` (so winner -> ``(beta_j - c) a``). Idle-
        allowed blocks (incl. input/rejection rows) stay at the nominal baseline
        when no activity scores positive -- i.e. the server idles / the input
        rejects. This recovers the per-network behaviour-policy allocation from
        the generic block structure.
        """
        g = np.atleast_2d(np.asarray(g, dtype=np.float64))
        B = g.shape[0]
        alpha = g @ self.R + self.index_offset            # (B, J)
        beta = self.params.nominal_allocation
        theta = np.zeros((B, self.spec.num_activities))
        ar = np.arange(B)
        for k in range(self.spec.num_resources):
            idx = self.spec.resource_block(k)
            if idx.size == 0:
                continue
            beta_blk = beta[idx]
            c = float(beta_blk.sum())
            a_blk = alpha[:, idx]                          # (B, |idx|)
            theta[:, idx] = beta_blk * a                   # nominal baseline
            best = np.argmax(a_blk, axis=1)                # (B,)
            best_val = a_blk[ar, best]
            acts = (best_val > 0.0) if self.spec.idle_allowed[k] else np.ones(B, bool)
            theta[ar, idx[best]] -= c * a * acts           # winner takes capacity
        return theta

    def policy_ref_drift(self, g: np.ndarray, a: float) -> np.ndarray:
        """On-policy reference state-drift mu = zeta + theta @ R.T  (B, I)."""
        theta = self.extract_theta(g, a)
        return self.zeta + theta @ self.R.T

    def summary(self) -> dict:
        return {
            "name": self.spec.name,
            "zeta": self.zeta.tolist(),
            "Gamma": self.Gamma.tolist(),
            "R_shape": list(self.R.shape),
            "K_shape": list(self.K_matrix.shape),
            "H": self.H.tolist(),
            "chi": self.chi.tolist(),
            "h_tilde": self.h_tilde.tolist(),
            "c_tilde": self.c_tilde.tolist(),
            "input_opportunity_rate": self.input_opportunity_rate.tolist(),
            "b": self.params.b,
            "gamma": self.params.gamma,
            "n": self.params.n,
        }
