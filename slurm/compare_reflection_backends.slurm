#!/bin/bash
#SBATCH --account=pi-bata0
#SBATCH --partition=standard_hopper
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=bcp_refcmp
#SBATCH --output=results/logs/bcp_refcmp-%j.out
#SBATCH --error=results/logs/bcp_refcmp-%j.err

set -euo pipefail

command -v module >/dev/null 2>&1 && module load python/booth/3.12 || true
export PATH="/apps/python/RHEL8/3.12/3.12.9/bin:$PATH"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "host=$(hostname)"
echo "date=$(date)"

python3 -u benchmarks/compare_reflection_backend_outputs.py
