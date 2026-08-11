"""NetworkSpec: the prelimit network primitives of math_foundation.md Section 1.

One immutable data model represents Criss-Cross, Pesic-Williams, and the
Three-Station Bigstep network. There is no per-network subclass; the scientific
content lives entirely in the arrays loaded from YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_float_array(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        raise ValueError(f"{name} is empty")
    return a


@dataclass(frozen=True)
class NetworkSpec:
    """Prelimit network N = (I, J, K, P, A, mu, h, c^I, c^D, rho, kappa, iota).

    Array conventions (math_foundation.md Section 1):
      buffer_change   P  (I x J)  queue increment per activity
      resource_use    A  (K x J)  resource consumption (A_kj > 0 => activity j uses k)
      rates           mu (J)      prelimit activity rates
      holding_cost    h  (I)
      input_cost      c^{I,(n)} (K)  prelimit per-occurrence processing-idleness cost (input rows zero)
      rejection_cost  c^{D,(n)} (K)  prelimit per-job rejected-job cost (input rows may be nonzero)
      discount        rho
      resource_types  kappa-coded strings: "processing" or "input"
      idle_allowed    iota (K)    True if idleness / no-admission is admissible
    """

    name: str
    buffer_change: np.ndarray
    resource_use: np.ndarray
    rates: np.ndarray
    holding_cost: np.ndarray
    input_cost: np.ndarray
    rejection_cost: np.ndarray
    discount: float
    resource_types: tuple[str, ...]
    idle_allowed: np.ndarray
    buffer_labels: tuple[str, ...] = field(default=())
    activity_labels: tuple[str, ...] = field(default=())
    resource_labels: tuple[str, ...] = field(default=())

    # ------------------------------------------------------------------ sizes
    @property
    def num_buffers(self) -> int:
        return self.buffer_change.shape[0]

    @property
    def num_activities(self) -> int:
        return self.buffer_change.shape[1]

    @property
    def num_resources(self) -> int:
        return self.resource_use.shape[0]

    # --------------------------------------------------------------- derived
    @property
    def consumption(self) -> np.ndarray:
        """C_ij = (-P_ij)^+  (jobs activity j consumes from buffer i)."""
        return np.maximum(-self.buffer_change, 0.0)

    @property
    def production(self) -> np.ndarray:
        """(P_ij)^+  (jobs activity j produces into buffer i)."""
        return np.maximum(self.buffer_change, 0.0)

    @property
    def is_processing(self) -> np.ndarray:
        return np.array([t == "processing" for t in self.resource_types], dtype=bool)

    @property
    def is_input(self) -> np.ndarray:
        return np.array([t == "input" for t in self.resource_types], dtype=bool)

    @property
    def processing_resources(self) -> np.ndarray:
        return np.where(self.is_processing)[0]

    @property
    def input_resources(self) -> np.ndarray:
        return np.where(self.is_input)[0]

    @property
    def no_rejection_resources(self) -> np.ndarray:
        """K_0: input rows with idle_allowed == False (A_0 Y = 0 equality rows)."""
        return np.where(self.is_input & ~self.idle_allowed)[0]

    def resource_block(self, k: int) -> np.ndarray:
        """B_k = {j : A_kj > 0}."""
        return np.where(self.resource_use[k] > 0)[0]

    @property
    def resource_blocks(self) -> list[np.ndarray]:
        return [self.resource_block(k) for k in range(self.num_resources)]

    def activity_resources(self, j: int) -> np.ndarray:
        """All resources used by activity j (A_kj > 0)."""
        return np.where(self.resource_use[:, j] > 0)[0]

    def activity_resource(self, j: int) -> int:
        """The single resource used by activity j; -1 if none.

        Returns the first resource if activity j uses several. Under the
        supported single-resource-per-activity scope (see
        validation.validate_supported_scope) this is unique; for a multi-resource
        activity the block policy is not valid and that scope check fails first.
        """
        rows = self.activity_resources(j)
        return int(rows[0]) if rows.size else -1

    def consumed_buffers(self, j: int) -> np.ndarray:
        """Buffers that activity j consumes (C_ij > 0)."""
        return np.where(self.consumption[:, j] > 0)[0]

    # ---------------------------------------------------------------- helpers
    def label_index(self, kind: str, label: str) -> int:
        labels = {
            "buffer": self.buffer_labels,
            "activity": self.activity_labels,
            "resource": self.resource_labels,
        }[kind]
        return labels.index(label)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "num_buffers": self.num_buffers,
            "num_activities": self.num_activities,
            "num_resources": self.num_resources,
            "processing_resources": self.processing_resources.tolist(),
            "input_resources": self.input_resources.tolist(),
            "no_rejection_resources": self.no_rejection_resources.tolist(),
            "discount": self.discount,
        }


def network_from_dict(d: dict) -> NetworkSpec:
    """Build a NetworkSpec from a parsed YAML/JSON mapping."""
    P = _as_float_array(d["buffer_change"], "buffer_change")
    A = _as_float_array(d["resource_use"], "resource_use")
    mu = _as_float_array(d["rates"], "rates")
    h = _as_float_array(d["holding_cost"], "holding_cost")
    I, Jp = P.shape
    Kp, Ja = A.shape

    def _vec(key, length, default=0.0):
        if key in d and d[key] is not None:
            return _as_float_array(d[key], key)
        return np.full(length, float(default))

    cI = _vec("input_cost", Kp)
    cD = _vec("rejection_cost", Kp)

    resource_types = tuple(str(t) for t in d["resource_types"])
    idle = np.asarray(d["idle_allowed"], dtype=bool)

    labels_b = tuple(str(x) for x in d.get("buffers", [f"b{i+1}" for i in range(I)]))
    labels_a = tuple(str(x) for x in d.get("activities", [f"j{j+1}" for j in range(Jp)]))
    labels_r = tuple(str(x) for x in d.get("resources", [f"k{k+1}" for k in range(Kp)]))

    return NetworkSpec(
        name=str(d["name"]),
        buffer_change=P,
        resource_use=A,
        rates=mu,
        holding_cost=h,
        input_cost=cI,
        rejection_cost=cD,
        discount=float(d["discount"]),
        resource_types=resource_types,
        idle_allowed=idle,
        buffer_labels=labels_b,
        activity_labels=labels_a,
        resource_labels=labels_r,
    )
