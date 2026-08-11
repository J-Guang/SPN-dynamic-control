#!/bin/bash
#SBATCH --account=pi-bata0
#SBATCH --partition=standard_hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=bcp_ref_cpw
#SBATCH --output=results/logs/bcp_ref_cpw-%A_%a.out
#SBATCH --error=results/logs/bcp_ref_cpw-%A_%a.err
#SBATCH --array=0-5

set -euo pipefail

command -v module >/dev/null 2>&1 && module load python/booth/3.12 || true
export PATH="/apps/python/RHEL8/3.12/3.12.9/bin:$PATH"

NCPU="${SLURM_CPUS_PER_TASK:-8}"
export NUMBA_NUM_THREADS="$NCPU" OMP_NUM_THREADS="$NCPU"
export OPENBLAS_NUM_THREADS="$NCPU" MKL_NUM_THREADS="$NCPU"
export TF_NUM_INTRAOP_THREADS="$NCPU" TF_NUM_INTEROP_THREADS="$NCPU"
export KMP_DUPLICATE_LIB_OK=TRUE

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
mkdir -p results/logs results/processed/reflection_backend_cc results/processed/reflection_backend_pw

NETWORKS=(crisscross crisscross crisscross pesic_williams pesic_williams pesic_williams)
CONFIGS=(configs/net_topology/crisscross.yaml configs/net_topology/crisscross.yaml configs/net_topology/crisscross.yaml configs/net_topology/pesic_williams.yaml configs/net_topology/pesic_williams.yaml configs/net_topology/pesic_williams.yaml)
OUTDIRS=(results/processed/reflection_backend_cc results/processed/reflection_backend_cc results/processed/reflection_backend_cc results/processed/reflection_backend_pw results/processed/reflection_backend_pw results/processed/reflection_backend_pw)
BACKENDS=(exact_enum lemke pgs exact_enum lemke pgs)

IDX="${SLURM_ARRAY_TASK_ID}"
NETWORK="${NETWORKS[$IDX]}"
CFG="${CONFIGS[$IDX]}"
OUTDIR="${OUTDIRS[$IDX]}"
BACKEND="${BACKENDS[$IDX]}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "network=${NETWORK}"
echo "backend=${BACKEND}"
echo "host=$(hostname)"
echo "date=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS}"

python3 -u benchmarks/benchmark_reflection_one_backend.py \
  --network "$CFG" \
  --backend "$BACKEND" \
  --out-dir "$OUTDIR" \
  --num-paths 256 \
  --num-steps 64 \
  --warmup-segments 1000 \
  --pool-segments 5000 \
  --seed 42
