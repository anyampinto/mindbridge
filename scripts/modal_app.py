# -*- coding: utf-8 -*-
"""
MindBridge — Modal (replaces Sherlock SLURM)

One-time setup:
  pip install modal
  modal setup
  modal secret create huggingface-secret HF_TOKEN=hf_...
  modal volume create mindbridge-data

Upload raw NSD data once (do this before running anything):
  modal volume put mindbridge-data /local/path/to/nsd_data    nsd_data
  modal volume put mindbridge-data /local/path/to/nsd_meta    nsd_meta
  # nsd_meta needs: nsd_expdesign.mat, nsd_stimuli.hdf5

Run:
  modal run modal_app.py                           # full pipeline, subj01
  modal run modal_app.py --subj subj02             # full pipeline, subj02
  modal run modal_app.py --mode ingest             # ingestion only
  modal run modal_app.py --mode train              # training only (needs cached targets)

Download results:
  modal volume get mindbridge-data checkpoints/subj01/best_stage1.pt .
  modal volume get mindbridge-data logs/subj01/stage1_loss_curve.png .
"""

import modal
from pathlib import Path

# =============================================================================
# VOLUME — persistent storage, replaces /scratch on Sherlock
# =============================================================================
volume = modal.Volume.from_name("mindbridge-data", create_if_missing=True)
MOUNT  = "/mnt/mindbridge"   # where the volume is mounted inside the container

# =============================================================================
# CONTAINER IMAGE
# =============================================================================
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libglib2.0-0", "libgl1")
    # Install CUDA-enabled torch first (cu121 matches Modal's GPU drivers)
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "open_clip_torch",
        "diffusers>=0.27.0",
        "transformers>=4.40.0",
        "accelerate",
        "nibabel",
        "h5py",
        "scipy",
        "Pillow",
        "numpy<2",
        "matplotlib",
        "huggingface_hub",
    )
    # Bundle your local scripts into the image so they can be imported
    .add_local_python_source("cs231n_data_ingestion", "train_mindbridge")
)

app = modal.App("mindbridge", image=image)

# =============================================================================
# INGESTION FUNCTION
# =============================================================================
@app.function(
    gpu="A10G",                    # 24 GB VRAM — sufficient for CLIP+DINO+VAE sequentially
    volumes={MOUNT: volume},
    timeout=4 * 3600,
    cpu=8,
    memory=65536,
    secrets=[modal.Secret.from_name("huggingface-secret")],  # injects HF_TOKEN
)
def ingest(subj: str = "subj01"):
    from cs231n_data_ingestion import run_ingestion
    run_ingestion(subj=subj, root=Path(MOUNT))
    volume.commit()   # flush writes so they persist in the volume


# =============================================================================
# TRAINING FUNCTION
# =============================================================================
@app.function(
    gpu="A10G",
    volumes={MOUNT: volume},
    timeout=12 * 3600,
    cpu=8,
    memory=65536,
)
def train(subj: str = "subj01"):
    from train_mindbridge import run_training
    run_training(subj=subj, root=Path(MOUNT))
    volume.commit()


# =============================================================================
# PIPELINE FUNCTION — ingest → train in a single container (no cold-start gap)
# =============================================================================
@app.function(
    gpu="A10G",
    volumes={MOUNT: volume},
    timeout=16 * 3600,
    cpu=8,
    memory=65536,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def pipeline(subj: str = "subj01"):
    root         = Path(MOUNT)
    full_targets = root / "nsd_meta" / f"targets_full_{subj}.npz"

    if not full_targets.exists():
        print(f"=== STEP 1: Ingestion ({subj}) ===")
        from cs231n_data_ingestion import run_ingestion
        run_ingestion(subj=subj, root=root)
        volume.commit()
    else:
        print(f"Targets already cached — skipping ingestion.")

    print(f"=== STEP 2: Training ({subj}) ===")
    from train_mindbridge import run_training
    run_training(subj=subj, root=root)
    volume.commit()


# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================
@app.local_entrypoint()
def main(subj: str = "subj01", mode: str = "pipeline"):
    if mode == "ingest":
        ingest.remote(subj=subj)
    elif mode == "train":
        train.remote(subj=subj)
    else:
        pipeline.remote(subj=subj)
