#!/usr/bin/env python3
"""Generate result tables from configs and processed outputs.

Always emits a network-constants table (from the YAML configs). If prelimit /
train / eval JSON summaries exist under results/processed/, a results table is
emitted too. Outputs Markdown + CSV under tables/generated/.
"""
from __future__ import annotations

import argparse
import glob
import os

import _bootstrap  # noqa: F401
import numpy as np

from diffusion_based_policy.bcp import BCPModel
from diffusion_based_policy.config import load_network
from diffusion_based_policy.io import ensure_dir, read_json, write_csv

ROOT = _bootstrap.publication_root()
PROCESSED = os.path.join(ROOT, "results", "processed")

NETWORKS = {
    "crisscross": "configs/net_topology/crisscross.yaml",
    "pesic_williams": "configs/net_topology/pesic_williams.yaml",
    "three_station_bigstep": "configs/net_topology/three_station_bigstep.yaml",
}


def network_table() -> tuple[str, list[dict]]:
    rows = []
    for name, path in NETWORKS.items():
        spec, params = load_network(os.path.join(ROOT, path))
        m = BCPModel(spec, params)
        rows.append({
            "network": name,
            "I": spec.num_buffers, "J": spec.num_activities, "K": spec.num_resources,
            "n": params.n, "b": params.b, "gamma": params.gamma,
            "zeta": np.array2string(m.zeta, precision=3),
            "diag_Gamma": np.array2string(np.diag(m.Gamma), precision=3),
            "c_tilde": np.array2string(m.c_tilde, precision=3),
        })
    md = ["# Network constants\n",
          "| network | I | J | K | n | b | gamma | zeta | diag(Gamma) | c_tilde |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['network']} | {r['I']} | {r['J']} | {r['K']} | {r['n']} | "
                  f"{r['b']} | {r['gamma']} | {r['zeta']} | {r['diag_Gamma']} | {r['c_tilde']} |")
    return "\n".join(md) + "\n", rows


def results_table() -> tuple[str, list[dict]]:
    rows = []
    for path in sorted(glob.glob(os.path.join(PROCESSED, "prelimit_*.json"))):
        try:
            d = read_json(path)
        except Exception:
            continue
        rows.append({
            "experiment": d.get("experiment", os.path.basename(path)),
            "policy": d.get("policy", "?"),
            "cost_mean": round(d.get("cost_mean", float("nan")), 3),
            "cost_scaled": round(d.get("cost_scaled_mean", float("nan")), 5),
            "holding": round(d.get("holding", float("nan")), 3),
            "rejection": round(d.get("rejection", float("nan")), 3),
            "utilization": np.array2string(np.array(d.get("utilization", [])), precision=3),
        })
    for path in sorted(glob.glob(os.path.join(PROCESSED, "train_*.json"))):
        try:
            d = read_json(path)
        except Exception:
            continue
        rows.append({
            "experiment": d.get("experiment", os.path.basename(path)),
            "policy": "bsde-train",
            "cost_mean": "", "cost_scaled": d.get("v0", ""),
            "holding": "", "rejection": "",
            "utilization": f"final_loss={d.get('final_loss')}",
        })
    if not rows:
        return "# Results\n\n_(no processed prelimit/train outputs found yet)_\n", rows
    md = ["# Results\n",
          "| experiment | policy | cost_mean | cost_scaled / v0 | holding | rejection | utilization/notes |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['experiment']} | {r['policy']} | {r['cost_mean']} | "
                  f"{r['cost_scaled']} | {r['holding']} | {r['rejection']} | {r['utilization']} |")
    return "\n".join(md) + "\n", rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "tables", "generated"))
    args = ap.parse_args()
    ensure_dir(args.out)

    net_md, net_rows = network_table()
    res_md, res_rows = results_table()

    with open(os.path.join(args.out, "networks.md"), "w") as f:
        f.write(net_md)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(res_md)
    write_csv(os.path.join(args.out, "networks.csv"), net_rows)
    if res_rows:
        write_csv(os.path.join(args.out, "results.csv"), res_rows)

    print(net_md)
    print(res_md)
    print(f"tables -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
