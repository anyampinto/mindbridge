#!/bin/bash
#SBATCH --job-name=mindbridge_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/users/ampinto/mindbridge/scripts/mindbridge_train_%j.out
#SBATCH --error=/scratch/users/ampinto/mindbridge/scripts/mindbridge_train_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

# --- ENVIRONMENT FIXES ---
# Wipe any system path leakage from the login node environment
unset PYTHONPATH

# Load a modern GCC compiler to satisfy modern package build/runtime requirements
module load gcc/12.1.0

# --- CACHE ENVIRONMENT VARIABLES ---
export TRANSFORMERS_CACHE=/scratch/users/ampinto/mindbridge/hf_cache
export HF_HOME=/scratch/users/ampinto/mindbridge/hf_cache
export TORCH_HOME=/scratch/users/ampinto/mindbridge/torch_cache

# --- EXECUTION ---
cd /scratch/users/ampinto/mindbridge/scripts

# Run cleanly within your uv-managed isolated virtual environment
uv run python train_mindbridge.py

echo "End: $(date)"