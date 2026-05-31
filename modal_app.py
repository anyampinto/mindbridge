"""MindBridge pipeline on Modal: ingestion -> training -> reconstruction."""

import os
import sys
from pathlib import Path

import modal

MINDBRIDGE_ROOT = "/mnt/mindbridge"
VOLUME_NAME = "mindbridge-data"
SCRIPTS_DIR = "/root/scripts"

app = modal.App("mindbridge")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("awscli")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "h5py",
        "nibabel",
        "scipy",
        "matplotlib",
        "pillow",
        "open-clip-torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
    )
    .add_local_dir("scripts", remote_path=SCRIPTS_DIR)
)


def _configure_env(subj: str = "subj01") -> None:
    os.environ["MINDBRIDGE_ROOT"] = MINDBRIDGE_ROOT
    os.environ["NSD_SUBJ"] = subj
    os.environ["TRANSFORMERS_CACHE"] = f"{MINDBRIDGE_ROOT}/hf_cache"
    os.environ["HF_HOME"] = f"{MINDBRIDGE_ROOT}/hf_cache"
    os.environ["TORCH_HOME"] = f"{MINDBRIDGE_ROOT}/torch_cache"

    sys.path.insert(0, SCRIPTS_DIR)
    from paths import ensure_runtime_dirs

    ensure_runtime_dirs(MINDBRIDGE_ROOT, subj)


def _run_script(script_name: str) -> None:
    import runpy

    runpy.run_path(f"{SCRIPTS_DIR}/{script_name}", run_name="__main__")


gpu_fn = "A100"


@app.function(
    image=image,
    timeout=60 * 60 * 24,
    volumes={MINDBRIDGE_ROOT: volume},
)
def download_data(
    subj: str = "subj01",
    sessions: str = "all",
    stimuli: bool = True,
    imagery: bool = True,
    all_subjects: bool = False,
):
    """Download NSD fMRI + stimulus imagery + NSD-Imagery betas via AWS CLI (CPU only)."""
    _configure_env(subj)
    os.environ["NSD_SESSIONS"] = sessions
    os.environ["NSD_DOWNLOAD_STIMULI"] = "1" if stimuli else "0"
    os.environ["NSD_DOWNLOAD_IMAGERY"] = "1" if imagery else "0"
    os.environ["NSD_ALL_SUBJECTS"] = "1" if all_subjects else "0"
    _run_script("download_nsd.py")
    volume.commit()
    return "Download complete for all subjects" if all_subjects else f"Download complete for {subj}"


@app.function(
    image=image,
    timeout=60 * 60 * 24,
    volumes={MINDBRIDGE_ROOT: volume},
)
def download_all(
    sessions: str = "all",
    stimuli: bool = True,
    imagery: bool = True,
):
    """Download full NSD + NSD-Imagery for subj01-subj08 with per-subject commits."""
    _configure_env("subj01")
    from download_nsd import (
        ALL_SUBJECTS,
        download_imagery_betas,
        download_imagery_metadata,
        download_perception_sessions,
        download_shared_metadata,
        parse_sessions,
        require_aws_cli,
    )

    require_aws_cli()
    root = Path(MINDBRIDGE_ROOT)

    download_shared_metadata(root, stimuli=stimuli)
    volume.commit()
    if imagery:
        download_imagery_metadata(root)
        volume.commit()

    for subj in ALL_SUBJECTS:
        print(f"\n>>> Downloading {subj}")
        subj_sessions = parse_sessions(sessions, subj=subj)
        download_perception_sessions(subj, root, sessions=subj_sessions)
        if imagery:
            download_imagery_betas(subj, root)
        volume.commit()
        print(f">>> Committed volume after {subj}")

    return f"Download complete for {', '.join(ALL_SUBJECTS)}"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 12, volumes={MINDBRIDGE_ROOT: volume})
def ingest(subj: str = "subj01", compute_targets: bool = False):
    """Flatten betas (default) or run full target ingestion for training."""
    _configure_env(subj)
    script = "cs231n_data_ingestion.py" if compute_targets else "ingest_betas.py"
    _run_script(script)
    volume.commit()
    return f"Ingestion complete for {subj}"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 24, volumes={MINDBRIDGE_ROOT: volume})
def train(subj: str = "subj01"):
    """Run MindBridge Stage 1 perception pretraining."""
    _configure_env(subj)
    from train_mindbridge_loss import run_training

    run_training(subj=subj, root=Path(MINDBRIDGE_ROOT))
    volume.commit()
    return f"Training complete for {subj}"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 6, volumes={MINDBRIDGE_ROOT: volume})
def reconstruct(subj: str = "subj01", n: int = 12, vd: bool = False):
    """Reconstruct images from a saved checkpoint."""
    _configure_env(subj)
    from reconstruct import main as reconstruct_main

    argv = ["reconstruct.py", "--subj", subj, "--root", MINDBRIDGE_ROOT, "--n", str(n)]
    if vd:
        argv.append("--vd")
    sys.argv = argv
    reconstruct_main()
    volume.commit()
    return f"Reconstruction complete for {subj}"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 24, volumes={MINDBRIDGE_ROOT: volume})
def run_pipeline(subj: str = "subj01"):
    """Full pipeline: ingestion (if needed) then Stage 1 training."""
    _configure_env(subj)
    targets = Path(MINDBRIDGE_ROOT) / "nsd_meta" / f"targets_full_{subj}.npz"
    if not targets.exists():
        _run_script("cs231n_data_ingestion.py")
        volume.commit()

    from train_mindbridge_loss import run_training

    run_training(subj=subj, root=Path(MINDBRIDGE_ROOT))
    volume.commit()
    return f"Pipeline complete for {subj}"


@app.function(
    image=image,
    timeout=60 * 10,
    volumes={MINDBRIDGE_ROOT: volume},
)
def check_nsd_volume():
    """Report local session counts on the volume and S3 availability for missing files."""
    _configure_env("subj01")
    _run_script("check_nsd_volume.py")
    return "check complete"


@app.local_entrypoint()
def main(
    subj: str = "subj01",
    step: str = "reconstruct",
    n: int = 12,
    vd: bool = False,
    sessions: str = "all",
    stimuli: bool = True,
    imagery: bool = True,
    all_subjects: bool = False,
    compute_targets: bool = False,
):
    """Run MindBridge on Modal.

    Examples:
      modal run modal_app.py --step download --all-subjects
      modal run modal_app.py --subj subj01 --step download
      modal run modal_app.py --subj subj01 --step ingest              # betas only
      modal run modal_app.py --subj subj01 --step ingest --compute-targets  # full
      modal run modal_app.py --subj subj01 --step reconstruct
    """
    if step == "download":
        if all_subjects:
            print(download_all.remote(sessions, stimuli, imagery))
        else:
            print(download_data.remote(subj, sessions, stimuli, imagery, False))
    elif step == "ingest":
        print(ingest.remote(subj, compute_targets))
    elif step == "train":
        print(train.remote(subj))
    elif step == "reconstruct":
        print(reconstruct.remote(subj, n, vd))
    elif step == "check":
        print(check_nsd_volume.remote())
    else:
        print(run_pipeline.remote(subj))
