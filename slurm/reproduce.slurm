#!/bin/bash
#SBATCH --account=pi-bata0
#SBATCH --partition=standard_hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=bcp_repro
#SBATCH --output=results/logs/bcp_repro-%A_%a.out
#SBATCH --error=results/logs/bcp_repro-%A_%a.err
#SBATCH --array=0-2
#
# Reproduce the three published networks with the default recipe:
#   b = 10, seed = 42, b-annealing from initial_bound = 0 (pace = 10 on every
#   network), offline BSDE training, numba LCP backend.
#
# All networks use hidden = [300, 300, 300]. Total steps follow the 1/4 annealing
# rule -- annealing occupies the first quarter of training, so total = 4 x
# (anneal-to-b duration):
#   Pesic-Williams   anneal 12k -> 48k steps
#   Three-Station 3x anneal 32k -> 128k steps  (I = 8, so the ramp is longer)
#   Criss-Cross      anneal 12k -> 48k steps
#
# Evaluation per network: a 100k-path prelimit policy run (6000 jumps, H = 3.0)
# and a 100k-path diffusion-scale V_MC, both reading their settings from config.
set -euo pipefail

command -v module >/dev/null 2>&1 && module load python/booth/3.12 || true
export PATH="/apps/python/RHEL8/3.12/3.12.9/bin:$PATH"
PY=python3
NCPU="${SLURM_CPUS_PER_TASK:-8}"
# Thread-pool caps, all pinned to the allocation. numba and TensorFlow SHARE one
# OpenMP runtime (KMP_DUPLICATE_LIB_OK), so capping TF_NUM_INTRAOP below NCPU also
# starves numba's parallel reflection -- keep every pool at NCPU so the pool build
# stays multi-core. The pools share the runtime, so this does not oversubscribe.
export NUMBA_NUM_THREADS="$NCPU" OMP_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export TF_NUM_INTRAOP_THREADS="$NCPU" TF_NUM_INTEROP_THREADS="$NCPU"
export KMP_DUPLICATE_LIB_OK=TRUE

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p results/logs results/processed results/checkpoints

NAMES=(bsde_pesic_williams bsde_bigstep bsde_crisscross)
STEPS=(48000             128000       48000)
i=${SLURM_ARRAY_TASK_ID:-0}
NAME=${NAMES[$i]}
S=${STEPS[$i]}
CFG=configs/net_parameters/${NAME}.yaml
TAG=${TAG:-_repro}            # override via: sbatch --export=ALL,TAG=_repro_h100 ...
CKPT=results/checkpoints/${NAME}${TAG}

echo "===== reproduce ${NAME}  steps=${S}  (b=10, anneal-from-0, offline) ====="
echo "host: $(hostname)  date: $(date)  GPU: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "============================================================================"

# 1) train: default b-curriculum (0 -> 10) and seed/b are taken from the config.
$PY cli/train_bsde.py --config "$CFG" --steps "$S" --tag "$TAG"

# 2) prelimit policy evaluation: 100k paths, 6000 jumps, H = 3.0 (config defaults).
$PY cli/eval_policy.py --config "$CFG" --tag "$TAG" --paths 100000

# 3) diffusion-scale value: 100k reflected-diffusion rollouts of the trained net.
$PY cli/eval_vmc.py --config "$CFG" --ckpt "$CKPT" --paths 100000 --tag "${TAG}_100k"

# 4) one-line summary tying the three artifacts together.
$PY - "$NAME" "$TAG" <<'PYEOF'
import json, sys
name, tag = sys.argv[1], sys.argv[2]
proc = "results/processed"
def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}
tr = load(f"{proc}/{name}{tag}/summary.json").get("result", {})
ev = load(f"{proc}/eval_{name}{tag}.json").get("prelimit", {})
vm = load(f"{proc}/vmc_{name}{tag}_100k.json")
print(f"\n=== {name} reproduction summary (b=10, seed=42) ===")
v0 = tr.get("v0")
print(f"  V_NN(0)        = {v0:.4f}" if v0 is not None else "  V_NN(0)        = <missing>")
vmc, se = vm.get("v_mc"), vm.get("v_mc_stderr")
print(f"  V_MC(0) 100k   = {vmc:.4f} +/- {se:.4f}" if vmc is not None
      else "  V_MC(0) 100k   = <missing>")
cm, hd, rj = ev.get("cost_mean"), ev.get("holding"), ev.get("rejection")
print(f"  prelimit cost  = {cm:.1f}  (holding {hd:.0f} + rejection {rj:.0f})"
      if cm is not None else "  prelimit cost  = <missing>")
PYEOF
echo "${NAME} reproduction complete"
