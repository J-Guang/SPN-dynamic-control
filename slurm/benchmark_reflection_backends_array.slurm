#!/bin/bash
#SBATCH --account=pi-bata0
#SBATCH --partition=standard_hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=bcp_refarr
#SBATCH --output=results/logs/bcp_refarr-%A_%a.out
#SBATCH --error=results/logs/bcp_refarr-%A_%a.err
#SBATCH --array=0-2

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
mkdir -p results/logs results/processed/reflection_backend_full

BACKENDS=(exact_enum lemke pgs)
BACKEND="${BACKENDS[$SLURM_ARRAY_TASK_ID]}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "backend=${BACKEND}"
echo "host=$(hostname)"
echo "date=$(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS}"

python3 -u benchmarks/benchmark_reflection_one_backend.py \
  --backend "$BACKEND" \
  --num-paths 256 \
  --num-steps 64 \
  --warmup-segments 1000 \
  --pool-segments 5000 \
  --seed 42
