#!/bin/bash
#SBATCH --job-name=mindbridge_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/ampinto/mindbridge/scripts/mindbridge_full_%j.out
#SBATCH --error=/scratch/users/ampinto/mindbridge/scripts/mindbridge_full_%j.err

echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "Start:     $(date)"
echo "========================================"

# =============================================================================
# ENVIRONMENT IMMUNIZATION (The Fixes)
# =============================================================================
# 1. Wipe global cluster path leaks completely
unset PYTHONPATH
unset PYTHONHOME

# 2. Load the verified Sherlock system compiler suite
module load gcc/12.4.0

# 3. Cache directory exports
export TRANSFORMERS_CACHE=/scratch/users/ampinto/mindbridge/hf_cache
export HF_HOME=/scratch/users/ampinto/mindbridge/hf_cache
export TORCH_HOME=/scratch/users/ampinto/mindbridge/torch_cache

# 4. Activate local isolated virtual environment
source /scratch/users/ampinto/mindbridge/.venv/bin/activate

# 5. Force the active path wrapper to prioritize your local .venv binaries
export PATH=/scratch/users/ampinto/mindbridge/.venv/bin:$PATH

# 6. Standardize relative context execution folder
cd /scratch/users/ampinto/mindbridge/scripts


# =============================================================================
# STEP 1: INGESTION
# Skip if targets already exist
# =============================================================================
TARGETS=/scratch/users/ampinto/mindbridge/nsd_meta/targets_session01.npz

if [ -f "$TARGETS" ]; then
    echo "Targets already exist — skipping ingestion."
else
    echo ""
    echo "=== STEP 1: Running ingestion ==="
    # Explicitly using the full local python path to protect environment execution
    /scratch/users/ampinto/mindbridge/.venv/bin/python cs231n_data_ingestion.py
    if [ $? -ne 0 ]; then
        echo "Ingestion failed. Exiting."
        exit 1
    fi
    echo "Ingestion complete."
fi


# =============================================================================
# STEP 2: STAGE 1 PRETRAINING (full run, no LoRA)
# =============================================================================
echo ""
echo "=== STEP 2: Running Stage 1 pretraining ==="
# Explicitly using the full local python path to protect environment execution
/scratch/users/ampinto/mindbridge/.venv/bin/python train_mindbridge.py
if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo ""
echo "========================================"
echo "All done: $(date)"
echo "========================================"