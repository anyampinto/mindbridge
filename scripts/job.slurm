#!/bin/bash
#SBATCH --job-name=mindbridge
#SBATCH --output=slurm-%j.out
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Edit partition, GPUs, and resources for your workload:
# #SBATCH --gres=gpu:1
# #SBATCH --partition=gpu

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load python/3.12.1
# Ensure uv is on PATH (adjust if you installed elsewhere)
export PATH="${HOME}/.local/bin:${PATH}"
source .venv/bin/activate

# Or: uv run python your_script.py
