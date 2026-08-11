#!/usr/bin/env python3
"""Load the three paper networks, validate them, and report BCP matrices.

Verifies the Section 11 checklist of math_foundation.md:
  * configs load into NetworkSpec
  * shape and sign validation
  * P diag(mu*) beta = 0
  * A beta = e on critical rows
  * zeta, Gamma, Q, H = R Q
  * Brownian rejection-cost scaling tilde c_k = mu_bar_k c^D_k

Writes a JSON summary to results/processed/check_networks.json.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# allow running from publication/ without installation
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from diffusion_based_policy.bcp import BCPModel  # noqa: E402
from diffusion_based_policy.config import load_network  # noqa: E402
from diffusion_based_policy.costs import brownian_rejection_costs, verify_rejection_scaling  # noqa: E402
from diffusion_based_policy.io import write_json  # noqa: E402
from diffusion_based_policy.planning import static_planning_lp, verify_critical_pair  # noqa: E402
from diffusion_based_policy.validation import (  # noqa: E402
    validate_bcp,
    validate_network_spec,
    validate_supported_scope,
)

NETWORKS = {
    "crisscross": "configs/net_topology/crisscross.yaml",
    "pesic_williams": "configs/net_topology/pesic_williams.yaml",
    "three_station_bigstep": "configs/net_topology/three_station_bigstep.yaml",
}


def check_one(name: str, path: str) -> dict:
    spec, params = load_network(path)
    model = BCPModel(spec, params)

    net_msgs = validate_network_spec(spec)
    bcp_msgs = validate_bcp(model)
    scope_msgs = validate_supported_scope(model)
    crit = verify_critical_pair(model)
    plan_star = static_planning_lp(
        spec, rates=params.critical_rates,
        pin_inputs=True, nominal_allocation=params.nominal_allocation,
    )
    plan_prelimit = static_planning_lp(spec)
    rej = verify_rejection_scaling(model)

    ok = (
        not net_msgs
        and not bcp_msgs
        and not scope_msgs
        and crit["flow_balance_ok"]
        and crit["critical_load_ok"]
        and crit["noncritical_proc_ok"]
        and crit["input_ok"]
        and rej["ok"]
    )

    print(f"\n=== {name} ===")
    print(f"  I={spec.num_buffers} J={spec.num_activities} K={spec.num_resources}")
    print(f"  network validation: {'OK' if not net_msgs else net_msgs}")
    print(f"  bcp validation:     {'OK' if not bcp_msgs else bcp_msgs}")
    print(f"  supported scope:    {'OK' if not scope_msgs else scope_msgs}")
    print(f"  flow balance |P diag(mu*) beta|_inf = "
          f"{np.max(np.abs(crit['flow_balance_residual'])):.2e}  -> {crit['flow_balance_ok']}")
    print(f"  critical load A beta: {crit['critical_load']}  -> {crit['critical_load_ok']}")
    print(f"  rho* (critical rates)  = {plan_star['rho_star']:.4f}")
    print(f"  rho* (prelimit rates)  = {plan_prelimit['rho_star']:.4f}")
    print(f"  zeta = {np.array2string(model.zeta, precision=4)}")
    print(f"  diag(Gamma) = {np.array2string(np.diag(model.Gamma), precision=4)}")
    print(f"  c_tilde (K rows) = {np.array2string(model.c_tilde, precision=4)}")
    print(f"  rejection scaling OK: {rej['ok']}")
    print(f"  H diag = {np.array2string(np.diag(model.H), precision=4)}")
    print(f"  >>> {name}: {'PASS' if ok else 'FAIL'}")

    return {
        "name": name,
        "ok": bool(ok),
        "sizes": {"I": spec.num_buffers, "J": spec.num_activities, "K": spec.num_resources},
        "network_validation": net_msgs,
        "bcp_validation": bcp_msgs,
        "supported_scope": scope_msgs,
        "critical_pair": {
            "flow_balance_inf_norm": float(np.max(np.abs(crit["flow_balance_residual"]))),
            "flow_balance_ok": crit["flow_balance_ok"],
            "critical_load": crit["critical_load"],
            "critical_load_ok": crit["critical_load_ok"],
            "input_ok": crit["input_ok"],
        },
        "planning": {
            "rho_star_critical": plan_star["rho_star"],
            "rho_star_prelimit": plan_prelimit["rho_star"],
        },
        "bcp": model.summary(),
        "rejection_costs": brownian_rejection_costs(model),
        "rejection_scaling_ok": rej["ok"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/processed/check_networks.json")
    args = ap.parse_args()

    results = {}
    all_ok = True
    for name, path in NETWORKS.items():
        if not os.path.exists(path):
            print(f"MISSING config: {path}")
            all_ok = False
            continue
        r = check_one(name, path)
        results[name] = r
        all_ok = all_ok and r["ok"]

    write_json(args.out, {"all_ok": all_ok, "networks": results})
    print(f"\nSummary written to {args.out}")
    print("ALL NETWORKS PASS" if all_ok else "SOME NETWORKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
