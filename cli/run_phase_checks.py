#!/usr/bin/env python3
"""Single entry point for end-to-end phase validation.

    python cli/run_phase_checks.py --phase all [--smoke]
    python cli/run_phase_checks.py --phase foundation|reflection|diffusion|bsde|policy

Each phase writes a compact JSON summary to results/processed/ and returns a
nonzero exit code on failure. ``--smoke`` keeps every phase tiny (small batches,
few steps) so the whole pipeline runs in well under a minute on CPU.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback

import _bootstrap  # noqa: F401
import numpy as np

from diffusion_based_policy.io import read_json, write_json

ROOT = _bootstrap.publication_root()
PROCESSED = os.path.join(ROOT, "results", "processed")

NETWORKS = {
    "crisscross": os.path.join(ROOT, "configs/net_topology/crisscross.yaml"),
    "pesic_williams": os.path.join(ROOT, "configs/net_topology/pesic_williams.yaml"),
    "three_station_bigstep": os.path.join(ROOT, "configs/net_topology/three_station_bigstep.yaml"),
}
EXPERIMENTS = {
    "crisscross": os.path.join(ROOT, "configs/net_parameters/bsde_crisscross.yaml"),
    "pesic_williams": os.path.join(ROOT, "configs/net_parameters/bsde_pesic_williams.yaml"),
    "three_station_bigstep": os.path.join(ROOT, "configs/net_parameters/bsde_bigstep.yaml"),
}


# --------------------------------------------------------------------------- 0
def phase_foundation(smoke: bool) -> dict:
    from diffusion_based_policy.bcp import BCPModel
    from diffusion_based_policy.config import load_network
    from diffusion_based_policy.costs import verify_rejection_scaling
    from diffusion_based_policy.planning import static_planning_lp, verify_critical_pair
    from diffusion_based_policy.validation import validate_bcp, validate_network_spec

    out = {}
    ok_all = True
    for name, path in NETWORKS.items():
        spec, params = load_network(path)
        model = BCPModel(spec, params)
        crit = verify_critical_pair(model)
        rej = verify_rejection_scaling(model)
        plan = static_planning_lp(
            spec, rates=params.critical_rates,
            pin_inputs=True, nominal_allocation=params.nominal_allocation,
        )
        ok = (
            not validate_network_spec(spec)
            and not validate_bcp(model)
            and crit["flow_balance_ok"]
            and crit["critical_load_ok"]
            and rej["ok"]
            and abs(plan["rho_star"] - 1.0) < 1e-6
        )
        ok_all = ok_all and ok
        out[name] = {"ok": bool(ok), "rho_star": plan["rho_star"],
                     "flow_balance_ok": crit["flow_balance_ok"],
                     "rejection_scaling_ok": rej["ok"]}
    return {"ok": bool(ok_all), "networks": out}


# --------------------------------------------------------------------------- 1
def phase_reflection(smoke: bool) -> dict:
    from diffusion_based_policy.bcp import BCPModel
    from diffusion_based_policy.config import load_network
    from diffusion_based_policy.reflection import available_backends, get_solver

    backends = available_backends()
    out = {"backends": backends, "networks": {}}
    ok_all = True
    rng = np.random.default_rng(0)
    n_paths = 256 if smoke else 4096
    for name, path in NETWORKS.items():
        spec, params = load_network(path)
        model = BCPModel(spec, params)
        H = model.H
        X = rng.standard_normal((n_paths, model.I)) * 0.5
        solver = get_solver("numba", H)
        res = solver.solve(X)
        lcp_ok = res.lcp_residual < 1e-6
        agree = {}
        if "gpu" in backends:
            gsolver = get_solver("gpu", H)
            gres = gsolver.solve(X)
            diff = float(np.max(np.abs(gres.reflected - res.reflected)))
            agree["gpu_vs_numba_max_diff"] = diff
            agree["gpu_vs_numba_ok"] = diff < 1e-6
            lcp_ok = lcp_ok and agree["gpu_vs_numba_ok"]
        ok_all = ok_all and lcp_ok
        out["networks"][name] = {"ok": bool(lcp_ok),
                                 "lcp_residual": float(res.lcp_residual),
                                 **agree}
    out["ok"] = bool(ok_all)
    return out


# --------------------------------------------------------------------------- 2
def phase_diffusion(smoke: bool) -> dict:
    from diffusion_based_policy.config import load_experiment
    from diffusion_based_policy.sim.diffusion import simulate_reference_paths
    from diffusion_based_policy.bsde.dataset import make_validation_batch

    out = {"networks": {}}
    ok_all = True
    n_paths = 64 if smoke else 1024
    n_steps = 16 if smoke else 64
    for name, exp_path in EXPERIMENTS.items():
        exp = load_experiment(exp_path)
        model = exp.model()
        bsde = exp.bsde or {}
        horizon = float(bsde.get("smoke_horizon", 0.1)) if smoke else float(bsde.get("horizon", 3.0))
        num_steps = int(bsde.get("smoke_steps", n_steps)) if smoke else int(bsde.get("num_steps", n_steps))
        ref_drift = bsde.get("reference_drift", None)
        paths = simulate_reference_paths(
            model, num_paths=n_paths, num_steps=num_steps,
            horizon=horizon, seed=exp.seed, backend="numba",
            reference_drift=ref_drift,
        )
        z = paths["z_path"]
        nonneg = bool(np.all(z >= -1e-6))
        finite = bool(np.all(np.isfinite(z)))
        batch = make_validation_batch(model, num_paths=n_paths, num_steps=num_steps,
                                      horizon=horizon, seed=exp.seed,
                                      backend="numba", reference_drift=ref_drift)
        shape_ok = batch["z"].shape[0] == n_paths
        ok = nonneg and finite and shape_ok
        ok_all = ok_all and ok
        out["networks"][name] = {"ok": bool(ok), "nonneg": nonneg,
                                 "finite": finite, "z_shape": list(z.shape)}
    out["ok"] = bool(ok_all)
    return out


# --------------------------------------------------------------------------- 3
def phase_bsde(smoke: bool) -> dict:
    from diffusion_based_policy.config import load_experiment
    from diffusion_based_policy.bsde.trainer import smoke_train

    out = {"networks": {}}
    ok_all = True
    for name, exp_path in EXPERIMENTS.items():
        exp = load_experiment(exp_path)
        result = smoke_train(exp, steps=20 if smoke else 200)
        ok = np.isfinite(result["final_loss"]) and result["final_loss"] >= 0
        ok_all = ok_all and ok
        out["networks"][name] = {"ok": bool(ok),
                                 "final_loss": float(result["final_loss"]),
                                 "v0": float(result["v0"])}
    out["ok"] = bool(ok_all)
    return out


# --------------------------------------------------------------------------- 4
def phase_policy(smoke: bool) -> dict:
    from diffusion_based_policy.config import load_experiment
    from diffusion_based_policy.policies.baselines import baseline_gradient
    from diffusion_based_policy.sim.prelimit import simulate_prelimit

    out = {"networks": {}}
    ok_all = True
    n_paths = 64 if smoke else 2000
    horizon = 0.5 if smoke else 3.0
    for name, exp_path in EXPERIMENTS.items():
        exp = load_experiment(exp_path)
        model = exp.model()
        grad_fn = baseline_gradient(model)
        res = simulate_prelimit(model, grad_fn, num_paths=n_paths,
                                horizon=horizon, seed=exp.seed)
        ok = np.isfinite(res["cost_mean"]) and res["cost_mean"] >= 0
        ok_all = ok_all and ok
        out["networks"][name] = {"ok": bool(ok),
                                 "cost_mean": float(res["cost_mean"]),
                                 "holding": float(res["holding"]),
                                 "rejection": float(res["rejection"]),
                                 "utilization": [float(u) for u in res["utilization"]]}
    out["ok"] = bool(ok_all)
    return out


PHASES = {
    "foundation": phase_foundation,
    "reflection": phase_reflection,
    "diffusion": phase_diffusion,
    "bsde": phase_bsde,
    "policy": phase_policy,
}
ORDER = ["foundation", "reflection", "diffusion", "bsde", "policy"]


def run_phase(name: str, smoke: bool) -> dict:
    t0 = time.time()
    try:
        result = PHASES[name](smoke)
        result["elapsed_sec"] = round(time.time() - t0, 3)
        return result
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e),
                "traceback": traceback.format_exc(),
                "elapsed_sec": round(time.time() - t0, 3)}


def run_all_isolated(smoke: bool) -> dict:
    """Run every phase in its own subprocess.

    TensorFlow and numba do not always coexist cleanly across many phases in a
    single long-lived process; isolating each phase in a fresh interpreter makes
    the end-to-end pipeline robust (and mirrors how the SLURM jobs run phases).
    """
    summary = {"phase": "all", "smoke": smoke, "results": {}}
    all_ok = True
    for name in ORDER:
        print(f"\n===== phase: {name} (smoke={smoke}) [subprocess] =====", flush=True)
        tmp = os.path.join(PROCESSED, f"_phase_{name}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--phase", name, "--out", tmp]
        if smoke:
            cmd.append("--smoke")
        proc = subprocess.run(cmd)
        if proc.returncode == 0 and os.path.exists(tmp):
            result = read_json(tmp)["results"][name]
        else:
            result = {"ok": False, "error": f"subprocess exit {proc.returncode}"}
        summary["results"][name] = result
        ok = result.get("ok", False)
        all_ok = all_ok and ok
        print(f"  -> {'PASS' if ok else 'FAIL'}  ({result.get('elapsed_sec')}s)")
    summary["all_ok"] = all_ok
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", *PHASES.keys()])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.phase == "all":
        summary = run_all_isolated(args.smoke)
        all_ok = summary["all_ok"]
    else:
        print(f"\n===== phase: {args.phase} (smoke={args.smoke}) =====", flush=True)
        result = run_phase(args.phase, args.smoke)
        ok = result.get("ok", False)
        all_ok = ok
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        print(f"  -> {'PASS' if ok else 'FAIL'}  ({result.get('elapsed_sec')}s)")
        summary = {"phase": args.phase, "smoke": args.smoke,
                   "results": {args.phase: result}, "all_ok": all_ok}

    out = args.out or os.path.join(PROCESSED, f"phase_checks_{args.phase}.json")
    write_json(out, summary)
    print(f"\nSummary -> {out}")
    print("ALL PHASES PASS" if all_ok else "SOME PHASES FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
