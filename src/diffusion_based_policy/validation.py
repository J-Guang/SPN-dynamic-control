"""Shape, sign, and structural validation for NetworkSpec and BCPModel."""
from __future__ import annotations

import numpy as np

from .bcp import BCPModel
from .network import NetworkSpec


class ValidationError(ValueError):
    pass


def _is_p_matrix(M: np.ndarray, tol: float = 1e-9) -> bool:
    """True iff every principal minor of M is (numerically) positive.

    P-matrix is the Harrison-Reiman condition for a valid Skorokhod reflection.
    For the dimensions here (I <= 8) the 2^I principal minors are cheap. Genuine
    minors of these reflection matrices are tiny but strictly positive (down to
    ~1e-10 for the 8x8 bigstep H), so we only reject minors that are clearly
    nonpositive (below a small negative floor), which catches singular / non-P
    matrices without false-rejecting well-posed ones.
    """
    from itertools import combinations

    n = M.shape[0]
    if n == 0:
        return True
    neg_floor = -1e-12
    for r in range(1, n + 1):
        for idx in combinations(range(n), r):
            if np.linalg.det(M[np.ix_(idx, idx)]) <= neg_floor:
                return False
    return True


def validate_network_spec(spec: NetworkSpec) -> list[str]:
    """Return a list of human-readable validation messages (empty == clean)."""
    msgs: list[str] = []
    I, J = spec.buffer_change.shape
    K = spec.resource_use.shape[0]

    if spec.resource_use.shape[1] != J:
        msgs.append(f"resource_use has {spec.resource_use.shape[1]} cols, expected {J}")
    if spec.rates.shape != (J,):
        msgs.append(f"rates shape {spec.rates.shape}, expected ({J},)")
    if spec.holding_cost.shape != (I,):
        msgs.append(f"holding_cost shape {spec.holding_cost.shape}, expected ({I},)")
    if spec.input_cost.shape != (K,):
        msgs.append(f"input_cost shape {spec.input_cost.shape}, expected ({K},)")
    if spec.rejection_cost.shape != (K,):
        msgs.append(f"rejection_cost shape {spec.rejection_cost.shape}, expected ({K},)")
    if len(spec.resource_types) != K:
        msgs.append(f"resource_types len {len(spec.resource_types)}, expected {K}")
    if spec.idle_allowed.shape != (K,):
        msgs.append(f"idle_allowed shape {spec.idle_allowed.shape}, expected ({K},)")

    if np.any(spec.rates < 0):
        msgs.append("rates must be nonnegative")
    if np.any(spec.holding_cost < 0):
        msgs.append("holding_cost must be nonnegative")
    if np.any(spec.resource_use < 0):
        msgs.append("resource_use must be nonnegative")
    if np.any(spec.rejection_cost < 0):
        msgs.append("rejection_cost must be nonnegative")
    if spec.discount <= 0:
        msgs.append("discount rho must be positive")

    for t in spec.resource_types:
        if t not in ("processing", "input"):
            msgs.append(f"resource_types entry '{t}' must be 'processing' or 'input'")

    # Processing-server idleness cost lives in input_cost; rejection cost must be
    # zero on processing rows and may be nonzero only on input rows.
    for k in range(K):
        if spec.is_processing[k] and spec.rejection_cost[k] != 0:
            msgs.append(f"rejection_cost nonzero on processing row {k}")
        if spec.is_input[k] and spec.input_cost[k] != 0:
            msgs.append(f"input_cost (processing idleness) nonzero on input row {k}")

    # Note: the single-resource-per-activity structural assumption is checked
    # separately by validate_supported_scope (it is a property of the supported
    # policy class, not of the generic NetworkSpec data model).
    return msgs


def validate_bcp(model: BCPModel, atol: float = 1e-9) -> list[str]:
    """Structural checks for the modified BCP (Section 3.1 universal checks)."""
    msgs: list[str] = []
    spec = model.spec

    # K Q >= 0
    KQ = model.K_matrix @ model.Q
    if KQ.size and np.min(KQ) < -atol:
        msgs.append(f"K Q has negative entries (min {float(np.min(KQ)):.3e})")

    # A_k Q = 0 for no-rejection input rows
    for k in spec.no_rejection_resources:
        row = spec.resource_use[k] @ model.Q
        if np.max(np.abs(row)) > atol:
            msgs.append(f"A_{spec.resource_labels[k]} Q != 0 (no-rejection input)")

    # H = R Q consistency
    if not np.allclose(model.H, model.R @ model.Q, atol=atol):
        msgs.append("H != R Q")

    # H should be a valid reflection matrix. The Harrison-Reiman condition is
    # that H is a P-matrix (all principal minors > 0); this both implies
    # nonsingularity and guarantees a well-defined Skorokhod map. Absolute det
    # is meaningless here (an 8x8 H with ~1/4 entries has det ~ 1e-10 yet is a
    # perfectly good P-matrix), so we test the P-matrix property directly.
    if np.any(np.diag(model.H) <= 0):
        msgs.append("H has nonpositive diagonal entries")
    if not _is_p_matrix(model.H, tol=atol):
        msgs.append("H is not a P-matrix (invalid reflection matrix)")

    # Gamma symmetric PSD
    if not np.allclose(model.Gamma, model.Gamma.T, atol=atol):
        msgs.append("Gamma not symmetric")
    eig = np.linalg.eigvalsh(0.5 * (model.Gamma + model.Gamma.T))
    if np.min(eig) < -atol:
        msgs.append(f"Gamma not PSD (min eig {float(np.min(eig)):.3e})")

    return msgs


def validate_supported_scope(model: BCPModel) -> list[str]:
    """Check the structural assumptions the policy / prelimit simulator rely on.

    The block-decomposed policy (policies/vectorized.py, policies/lifting.py) and
    the prelimit simulator assume the structural class of the three paper
    networks:

      * every activity uses exactly one resource (single-resource-per-activity),
        so resource blocks partition the activities and ``activity_resource`` is
        well defined;
      * every no-rejection input row (idle forbidden) has exactly one activity
        with unit consumption and beta = 1, so the lifted action can satisfy
        A_0 a = A_0 beta by simply admitting that activity.

    Networks outside this class need the general LP/MILP policy path; this
    function flags them rather than letting the block rule silently misbehave.
    """
    msgs: list[str] = []
    spec = model.spec

    used = (spec.resource_use > 0).sum(axis=0)
    multi = np.where(used > 1)[0]
    if multi.size:
        labels = [spec.activity_labels[j] for j in multi]
        msgs.append(f"activities using multiple resources (unsupported): {labels}")

    for k in spec.no_rejection_resources:
        block = spec.resource_block(k)
        basic = [j for j in block if model.params.nominal_allocation[j] > 0]
        if len(basic) != 1:
            msgs.append(
                f"no-rejection input '{spec.resource_labels[k]}' has "
                f"{len(basic)} basic activities (only single-activity supported)")
            continue
        j = basic[0]
        if abs(spec.resource_use[k, j] - 1.0) > 1e-9 or \
                abs(model.params.nominal_allocation[j] - 1.0) > 1e-9:
            msgs.append(
                f"no-rejection input '{spec.resource_labels[k]}' activity "
                f"'{spec.activity_labels[j]}' is not unit (A or beta != 1)")
    return msgs


def require_single_resource_per_activity(spec: NetworkSpec) -> None:
    """Fail fast unless every activity uses exactly one resource.

    The block-decomposed policy and Hamiltonian (policies/vectorized.py and
    bsde/losses.py) rely on resource blocks PARTITIONING the activities: only then
    is the per-resource argmax the exact LP optimum (math_foundation 3.5 / 5.4). An
    activity that consumes several resources couples the blocks, so the policy would
    need the general LP/MILP path -- not implemented in this release. We raise
    NotImplementedError here rather than let the block rule silently misbehave.
    """
    used = (spec.resource_use > 0).sum(axis=0)
    multi = np.where(used > 1)[0]
    if multi.size:
        labels = [spec.activity_labels[j] if spec.activity_labels else int(j)
                  for j in multi]
        raise NotImplementedError(
            "block-decomposed BCP policy requires single-resource-per-activity, but "
            f"these activities use multiple resources: {labels}. Multi-resource "
            "activities need the general LP/MILP policy path, which is not "
            "implemented in this release.")


def assert_valid(spec: NetworkSpec, model: BCPModel | None = None,
                 strict_scope: bool = False) -> None:
    msgs = validate_network_spec(spec)
    if model is not None:
        msgs += validate_bcp(model)
        if strict_scope:
            msgs += validate_supported_scope(model)
    if msgs:
        raise ValidationError("; ".join(msgs))
