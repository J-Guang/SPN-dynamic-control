"""Vectorized lifted policy for the prelimit simulator.

Computes, for a batch of integer queue states, the activity selected on each
resource (or -1 = idle/reject) under the lifted BCP index rule of
math_foundation.md Section 5.4. Fully numpy-vectorized so the simulator can run
tens of thousands of paths.

The independent per-resource block argmax is the global optimum whenever it does
not over-allocate a buffer (always the case for crisscross and bigstep, where no
buffer is shared across resources). For networks with a shared buffer
(Pesic-Williams b1) the block result can violate C a <= q; those (rare) rows are
re-solved exactly by enumerating the small set of executable action vertices --
no per-path LP, so it stays fast.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..bcp import BCPModel


class VectorizedPolicy:
    def __init__(self, model: BCPModel, work_conserving: bool = False):
        from ..validation import require_single_resource_per_activity
        require_single_resource_per_activity(model.spec)
        self.model = model
        spec = model.spec
        # If True, processing resources never idle: a station with any servable
        # queue always serves its best-index activity, even when that index is
        # <= 0 (classical work-conserving service). Input resources keep their
        # admission idle/reject behaviour regardless.
        self.work_conserving = bool(work_conserving)
        self.R = model.R                              # (I, J)
        self.offset = model.index_offset              # (J,)
        self.b = model.params.b
        self.C = spec.consumption                     # (I, J)
        self.P = spec.buffer_change                   # (I, J)
        self.K = spec.num_resources
        self.I = spec.num_buffers
        self.J = spec.num_activities
        self.beta = model.params.nominal_allocation
        self.idle_allowed = spec.idle_allowed
        self.no_rej = set(int(k) for k in spec.no_rejection_resources)

        self.blocks = []
        for k in range(self.K):
            idx = spec.resource_block(k)
            self.blocks.append({
                "idx": idx,
                "C": self.C[:, idx] if idx.size else np.zeros((self.I, 0)),
                "idle": bool(spec.idle_allowed[k]),
                "no_rej": k in self.no_rej,
                "processing": bool(spec.is_processing[k]),
            } if idx.size else None)

        self._shared = self._has_shared_buffer()
        if self._shared:
            self._build_candidates()

    # ------------------------------------------------------------- setup
    def _has_shared_buffer(self) -> bool:
        for i in range(self.I):
            acts = np.where(self.C[i] > 0)[0]
            res = {self.model.spec.activity_resource(int(j)) for j in acts}
            if len(res) > 1:
                return True
        return False

    def _build_candidates(self) -> None:
        """Enumerate executable action vertices (per-resource choice incl. idle)."""
        options = []
        for k, blk in enumerate(self.blocks):
            if blk is None:
                options.append([-1])
            elif blk["no_rej"]:
                # fixed: serve the nominal input activity(ies)
                served = [int(j) for j in blk["idx"] if self.beta[j] > 0]
                options.append([served[0]] if served else [-1])
            else:
                opts = [int(j) for j in blk["idx"]]
                opts.append(-1)  # idle / reject
                options.append(opts)
        combos = list(itertools.product(*options))
        ncomb = len(combos)
        CAND = np.array(combos, dtype=np.int64)            # (ncomb, K)
        ind = np.zeros((ncomb, self.J))                    # activity indicator
        ca = np.zeros((ncomb, self.I))                     # buffer consumption
        for c, combo in enumerate(combos):
            for k, j in enumerate(combo):
                if j >= 0:
                    ind[c, j] = 1.0
                    ca[c] += self.C[:, j]
        self._CAND = CAND
        self._CAND_ind = ind
        self._CAND_ca = ca

    # ------------------------------------------------------------- policy
    def policy_index(self, g: np.ndarray) -> np.ndarray:
        return self.b * (g @ self.R + self.offset)        # (M, J)

    def _block_argmax(self, q: np.ndarray, pi: np.ndarray) -> np.ndarray:
        M = q.shape[0]
        selected = np.full((M, self.K), -1, dtype=np.int64)
        for k, blk in enumerate(self.blocks):
            if blk is None:
                continue
            idx = blk["idx"]
            pib = pi[:, idx]
            feas = np.all(q[:, :, None] >= blk["C"][None, :, :] - 1e-9, axis=1)
            masked = np.where(feas, pib, -np.inf)
            best_local = np.argmax(masked, axis=1)
            best_val = masked[np.arange(M), best_local]
            chosen = idx[best_local]
            has_feasible = np.any(feas, axis=1)
            if self.work_conserving and blk["processing"]:
                admit = has_feasible                       # processing never idles
            elif blk["no_rej"]:
                admit = has_feasible
            elif blk["idle"]:
                admit = has_feasible & (best_val > 0.0)
            else:
                admit = has_feasible
            selected[:, k] = np.where(admit, chosen, -1)
        return selected

    def _buffer_usage(self, selected: np.ndarray) -> np.ndarray:
        M = selected.shape[0]
        U = np.zeros((M, self.I))
        for k in range(self.K):
            sel = selected[:, k]
            valid = sel >= 0
            if valid.any():
                U[valid] += self.C[:, sel[valid]].T
        return U

    def select(self, q: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Return selected[m, k] = chosen activity index for resource k, or -1."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        g = np.atleast_2d(np.asarray(g, dtype=np.float64))
        pi = self.policy_index(g)
        selected = self._block_argmax(q, pi)

        if not self._shared:
            return selected

        U = self._buffer_usage(selected)
        viol = np.where(np.any(U > q + 1e-9, axis=1))[0]
        if viol.size:
            qv = q[viol]                                   # (V, I)
            piv = pi[viol]                                 # (V, J)
            # objective per candidate: piv @ ind^T  (V, ncomb)
            obj = piv @ self._CAND_ind.T
            # feasibility: candidate consumption <= q  (V, ncomb)
            feas = np.all(self._CAND_ca[None, :, :] <= qv[:, None, :] + 1e-9, axis=2)
            masked = np.where(feas, obj, -np.inf)
            best = np.argmax(masked, axis=1)               # (V,)
            selected[viol] = self._CAND[best]
        return selected
