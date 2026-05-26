#!/bin/bash
#SBATCH --job-name=mindbridge_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/ampinto/mindbridge/scripts/mindbridge_full_%x_%j.out
#SBATCH --error=/scratch/users/ampinto/mindbridge/scripts/mindbridge_full_%x_%j.err

# Usage: sbatch run_full_pipeline.sbatch subj01
# If no argument given, defaults to subj01
export NSD_SUBJ=${1:-subj01}

echo "========================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Subject:  $NSD_SUBJ"
echo "Start:    $(date)"
echo "========================================"

unset PYTHONPATH
unset PYTHONHOME
module load gcc/12.4.0

export TRANSFORMERS_CACHE=/scratch/users/ampinto/mindbridge/hf_cache
export HF_HOME=/scratch/users/ampinto/mindbridge/hf_cache
export TORCH_HOME=/scratch/users/ampinto/mindbridge/torch_cache

source /scratch/users/ampinto/mindbridge/.venv/bin/activate
export PATH=/scratch/users/ampinto/mindbridge/.venv/bin:$PATH

cd /scratch/users/ampinto/mindbridge/scripts

# =============================================================================
# STEP 1: INGESTION — skip if full targets already exist for this subject
# =============================================================================
FULL_TARGETS=/scratch/users/ampinto/mindbridge/nsd_meta/targets_full_${NSD_SUBJ}.npz

if [ -f "$FULL_TARGETS" ]; then
    echo "Full targets already exist for $NSD_SUBJ — skipping ingestion."
else
    echo ""
    echo "=== STEP 1: Running ingestion for $NSD_SUBJ ==="
    python cs231n_data_ingestion.py
    if [ $? -ne 0 ]; then
        echo "Ingestion failed. Exiting."
        exit 1
    fi
    echo "Ingestion complete."
fi

# =============================================================================
# STEP 2: TRAINING
# =============================================================================
echo ""
echo "=== STEP 2: Running Stage 1 pretraining for $NSD_SUBJ ==="
python train_mindbridge.py
if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo ""
echo "========================================"
echo "All done: $(date)"
echo "========================================"
