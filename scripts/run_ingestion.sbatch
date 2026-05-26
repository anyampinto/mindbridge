#!/bin/bash
#SBATCH --job-name=mindbridge_ingest
#SBATCH --output=mindbridge_%j.out
#SBATCH --error=mindbridge_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --export=ALL

# Clean environment leaks
export PYTHONPATH=""
export PYTHONNOUSERSITE=1

# SSL bypass for local model weights
export TORCH_HUB_DISABLE_SECURE_RELOAD=1
export PYTHONHTTPSVERIFY=0

# Redirect caching structures to high-performance scratch
export HF_HOME="/scratch/users/ampinto/mindbridge/hf_cache"
export TORCH_HUB="/scratch/users/ampinto/mindbridge/torch_cache"

# 🔑 HUGGING FACE AUTHENTICATION
# Paste your actual token inside the quotes below
export HF_TOKEN="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"

echo "Checking allocated execution node hardware..."
nvidia-smi

echo "Initializing MindBridge Neural Pipeline..."
uv run python /scratch/users/ampinto/mindbridge/scripts/cs231n_data_ingestion.py

echo "Job pipeline finished successfully."