#!/bin/bash
#SBATCH --job-name=mindbridge_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=mindbridge_full_%x_%j.out
#SBATCH --error=mindbridge_full_%x_%j.err

# NSD_SUBJ can be passed via --export=ALL,NSD_SUBJ=subjXX or as $1
export NSD_SUBJ=${NSD_SUBJ:-${1:-subj01}}

echo "========================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Subject:  $NSD_SUBJ"
echo "Start:    $(date)"
echo "========================================"

unset PYTHONPATH
unset PYTHONHOME
module load gcc/12.4.0

export MINDBRIDGE_ROOT=${MINDBRIDGE_ROOT:-/scratch/users/ampinto/mindbridge}
export TRANSFORMERS_CACHE=$MINDBRIDGE_ROOT/hf_cache
export HF_HOME=$MINDBRIDGE_ROOT/hf_cache
export TORCH_HOME=$MINDBRIDGE_ROOT/torch_cache

source "$MINDBRIDGE_ROOT/.venv/bin/activate"
export PATH="$MINDBRIDGE_ROOT/.venv/bin:$PATH"

cd "$MINDBRIDGE_ROOT/scripts"

# STEP 0 (optional): download NSD + imagery from S3
# python download_nsd.py --subj $NSD_SUBJ
# Or on Modal: modal run modal_app.py --subj $NSD_SUBJ --step download

# =============================================================================
# STEP 1: INGESTION — skip if full targets already exist for this subject
# =============================================================================
FULL_TARGETS=$MINDBRIDGE_ROOT/nsd_meta/targets_full_${NSD_SUBJ}.npz

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
python train_mindbridge_loss.py
if [ $? -ne 0 ]; then
    echo "Training failed. Exiting."
    exit 1
fi

echo ""
echo "========================================"
echo "All done: $(date)"
echo "========================================"
