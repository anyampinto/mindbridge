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
        "scikit-learn",
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
    timeout=60 * 30,
    volumes={MINDBRIDGE_ROOT: volume},
)
def download_roi_all():
    """Sync full ROI mask directories for subj01-subj08."""
    _configure_env("subj01")
    from download_nsd import ALL_SUBJECTS, download_roi_masks

    root = Path(MINDBRIDGE_ROOT)
    for subj in ALL_SUBJECTS:
        download_roi_masks(subj, root)
        volume.commit()
        print(f">>> Committed ROI for {subj}")
    return f"ROI sync complete for {', '.join(ALL_SUBJECTS)}"


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


@app.function(
    image=image,
    timeout=60 * 60 * 2,
    volumes={MINDBRIDGE_ROOT: volume},
)
def download_imagery_meta(force: bool = False):
    """Sync NSD-Imagery design matrices and metadata from S3 (excludes *.mp4)."""
    _configure_env("subj01")
    from download_nsd import download_imagery_metadata

    download_imagery_metadata(Path(MINDBRIDGE_ROOT), force=force)
    volume.commit()
    return "NSD-Imagery metadata sync complete"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 12, volumes={MINDBRIDGE_ROOT: volume})
def ingest(subj: str = "subj01", compute_targets: bool = False):
    """Flatten betas (default) or run full target ingestion for training."""
    _configure_env(subj)
    script = "cs231n_data_ingestion.py" if compute_targets else "ingest_betas.py"
    _run_script(script)
    volume.commit()
    return f"Ingestion complete for {subj}"


@app.function(image=image, timeout=60 * 60 * 2, volumes={MINDBRIDGE_ROOT: volume})
def ingest_imagery(subj: str = "subj01"):
    """Ingest NSD-Imagery betas with trial metadata and per-stimulus averaging."""
    volume.reload()
    _configure_env(subj)
    from ingest_imagery_betas import run_ingestion

    summary = run_ingestion(subj=subj, root=Path(MINDBRIDGE_ROOT))
    volume.commit()
    return summary


@app.function(image=image, timeout=60 * 60 * 2, volumes={MINDBRIDGE_ROOT: volume})
def ingest_imagery_all():
    """Ingest NSD-Imagery betas for all 8 subjects in parallel."""
    results = list(ingest_imagery.map(ALL_SUBJECTS, return_exceptions=True))
    completed = {}
    failed = {}
    for subj, result in zip(ALL_SUBJECTS, results):
        if isinstance(result, Exception):
            failed[subj] = str(result)
        else:
            completed[subj] = result
    return {"completed": completed, "failed": failed}


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={MINDBRIDGE_ROOT: volume},
)
def ingest_averaged_cpu(subj: str = "all"):
    """Average perception betas/targets per image → nsd_meta/averaged/ (CPU only)."""
    volume.reload()
    _configure_env("subj01")
    import sys

    sys.argv = [
        "ingest_averaged.py",
        "--root",
        MINDBRIDGE_ROOT,
        "--subj",
        subj,
        "--skip-text",
    ]
    import runpy

    runpy.run_path(f"{SCRIPTS_DIR}/ingest_averaged.py", run_name="__main__")
    volume.commit()
    return "Averaged perception ingest (CPU) complete"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 8, volumes={MINDBRIDGE_ROOT: volume})
def ingest_averaged_text():
    """CLIP-text targets from COCO captions → nsd_meta/averaged/ (GPU)."""
    volume.reload()
    _configure_env("subj01")
    import sys

    sys.argv = [
        "ingest_averaged.py",
        "--root",
        MINDBRIDGE_ROOT,
        "--text-only",
        "--device",
        "cuda",
    ]
    import runpy

    runpy.run_path(f"{SCRIPTS_DIR}/ingest_averaged.py", run_name="__main__")
    volume.commit()
    return "Averaged CLIP-text ingest complete"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 12, volumes={MINDBRIDGE_ROOT: volume})
def ingest_clip_hidden(subj: str = "subj01"):
    """Compute 257×768 CLIP ViT hidden targets for averaged perception images."""
    volume.reload()
    _configure_env(subj)
    import sys
    from ingest_clip_hidden import main as ingest_main

    sys.argv = [
        "ingest_clip_hidden.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--batch-size", "16",
        "--device", "cuda",
    ]
    ingest_main()
    volume.commit()
    return f"clip_hidden ingest done for {subj}"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 24, volumes={MINDBRIDGE_ROOT: volume})
def train(
    subj: str = "subj01",
    script: str = "train_mindbridge_loss.py",
    epochs: int = 150,
    run_id: str = "",
    extra_argv: list[str] | None = None,
):
    """Run Stage 1 training (default or variant script). Outputs go to runs/{run_id}/."""
    volume.reload()
    _configure_env(subj)

    from paths import make_run_id

    if run_id:
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    elif not os.environ.get("MINDBRIDGE_RUN_ID"):
        label = f"{Path(script).stem}_{subj}_{epochs}ep"
        os.environ["MINDBRIDGE_RUN_ID"] = make_run_id(label)
    print(f">>> MINDBRIDGE_RUN_ID={os.environ['MINDBRIDGE_RUN_ID']}")

    if script == "train_mindbridge_loss.py":
        from train_mindbridge_loss import run_training

        run_training(subj=subj, root=Path(MINDBRIDGE_ROOT))
    else:
        sys.argv = [script, "--subj", subj, "--root", MINDBRIDGE_ROOT, "--epochs", str(epochs)]
        if extra_argv:
            sys.argv.extend(extra_argv)
        _run_script(script)

    volume.commit()
    rid = os.environ["MINDBRIDGE_RUN_ID"]
    return f"Training complete for {subj} ({script}, epochs={epochs}, run={rid})"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 24, volumes={MINDBRIDGE_ROOT: volume})
def train_all(subjects: list[str] | None = None):
    """Train subj02-subj08 in parallel; overwrites checkpoints/{subj}/best_stage1.pt."""
    subjects = subjects or RETRAIN_SUBJECTS
    completed = {}
    failed = {}
    for subj, result in zip(subjects, train.map(subjects, return_exceptions=True)):
        if isinstance(result, Exception):
            failed[subj] = str(result)
        else:
            completed[subj] = result
    return {"completed": completed, "failed": failed}


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 6, volumes={MINDBRIDGE_ROOT: volume})
def reconstruct_imagery_small(
    subj: str = "subj01",
    n: int = 3,
    strength: float = 0.6,
    run_id: str = "",
    ckpt_prefer: str = "final",
):
    """Reconstruct 3 Set-B imagery stimuli (Variant A checkpoint)."""
    volume.reload()
    _configure_env(subj)
    import sys
    from reconstruct_imagery_small import main as recon_main

    argv = [
        "reconstruct_imagery_small.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--n", str(n),
        "--strength", str(strength),
        "--ckpt-prefer", ckpt_prefer,
    ]
    if run_id:
        argv.extend(["--run-id", run_id])
    sys.argv = argv
    recon_main()
    volume.commit()
    rid = run_id or os.environ.get("MINDBRIDGE_RUN_ID", "?")
    return f"Imagery reconstruction saved for {subj} (run={rid})"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 6, volumes={MINDBRIDGE_ROOT: volume})
def reconstruct_imagery_4head(
    subj: str = "subj01",
    n: int = 3,
    strength: float = 0.6,
    run_id: str = "",
    variant: str = "4H",
    ckpt_prefer: str = "best_clip",
):
    """Reconstruct Set-B imagery with global 4H or 4head_roi checkpoint."""
    volume.reload()
    _configure_env(subj)
    import sys
    from reconstruct_imagery_4head import main as recon_main

    argv = [
        "reconstruct_imagery_4head.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--variant", variant,
        "--n", str(n),
        "--strength", str(strength),
        "--ckpt-prefer", ckpt_prefer,
    ]
    if run_id:
        argv.extend(["--run-id", run_id])
    sys.argv = argv
    recon_main()
    volume.commit()
    return f"Imagery 4-head recon ({variant}) saved for {subj} run={run_id}"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 6, volumes={MINDBRIDGE_ROOT: volume})
def reconstruct_perception_small(
    subj: str = "subj01",
    n: int = 3,
    strength: float = 0.6,
    run_id: str = "",
    ckpt_prefer: str = "final",
):
    """Reconstruct same Set-B NSD images from perception betas (Variant A checkpoint)."""
    volume.reload()
    _configure_env(subj)
    import sys
    from reconstruct_perception_small import main as recon_main

    if not run_id:
        raise ValueError("reconstruct-perception requires --run-id (training run with checkpoints)")

    sys.argv = [
        "reconstruct_perception_small.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--n", str(n),
        "--strength", str(strength),
        "--run-id", run_id,
        "--ckpt-prefer", ckpt_prefer,
    ]
    recon_main()
    volume.commit()
    return f"Perception reconstruction saved for {subj} (run={run_id})"


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


@app.function(
    image=image,
    timeout=60 * 10,
    volumes={MINDBRIDGE_ROOT: volume},
)
def probe_nsd_roi():
    """List ROI mask files on S3 vs the Modal volume."""
    _configure_env("subj01")
    _run_script("probe_nsd_roi.py")
    return "probe complete"


@app.function(
    image=image,
    timeout=60 * 5,
    volumes={MINDBRIDGE_ROOT: volume},
)
def inspect_roi_labels(subj: str = "subj01"):
    """Print labels and voxel counts for common ROI masks."""
    _configure_env(subj)
    _run_script("inspect_roi_labels.py")
    return "inspect complete"


@app.function(
    image=image,
    timeout=60 * 5,
    volumes={MINDBRIDGE_ROOT: volume},
)
def check_variant_ckpts(variant: str = "A"):
    """Print saved epoch for checkpoints_vX/subjXX/best.pt."""
    volume.reload()
    _configure_env("subj01")
    os.environ["MINDBRIDGE_VARIANT"] = variant.upper()
    _run_script("check_variant_ckpts.py")
    return f"check complete for variant {variant.upper()}"


@app.function(
    image=image,
    timeout=60 * 120,
    volumes={MINDBRIDGE_ROOT: volume},
)
def train_joint_ridge_subj(subj: str = "subj01", alpha: float = 1e4, run_id: str = ""):
    """Fit joint ridge baseline for one subject (CPU)."""
    volume.reload()
    _configure_env(subj)
    if run_id:
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    from train_joint_ridge import run_subject

    result = run_subject(subj, MINDBRIDGE_ROOT, alpha=alpha, run_id=run_id or None)
    volume.commit()
    return result


@app.function(
    image=image,
    timeout=60 * 120,
    volumes={MINDBRIDGE_ROOT: volume},
)
def train_joint_ridge_all(alpha: float = 1e4, run_id: str = ""):
    """Fit joint ridge for subj01-subj08 in parallel."""
    from datetime import datetime, timezone
    import json

    rid = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_joint_ridge_all8"
    print(f"=== Joint ridge run_id: {rid} ===")
    args = [(s, alpha, rid) for s in ALL_SUBJECTS]
    results = list(train_joint_ridge_subj.starmap(args))

    out_dir = Path(MINDBRIDGE_ROOT) / "runs" / rid / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "joint_ridge_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()
    return {"run_id": rid, "results": results, "out_file": str(out_file)}


ALL_SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
RETRAIN_SUBJECTS = [f"subj{i:02d}" for i in range(2, 9)]  # subj02-subj08


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 12, volumes={MINDBRIDGE_ROOT: volume})
def ingest_all(compute_targets: bool = True):
    """Run ingestion for all 8 subjects in parallel."""
    msgs = list(ingest.starmap([(subj, compute_targets) for subj in ALL_SUBJECTS]))
    return dict(zip(ALL_SUBJECTS, msgs))


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 2, volumes={MINDBRIDGE_ROOT: volume})
def compute_retrieval_subj(subj: str = "subj01"):
    """Compute 2-way retrieval + CLIP cosine similarity for one subject."""
    volume.reload()
    _configure_env(subj)
    from compute_retrieval import run_retrieval

    result = run_retrieval(subj, Path(MINDBRIDGE_ROOT))
    volume.commit()
    return result


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 2, volumes={MINDBRIDGE_ROOT: volume})
def compute_retrieval_all():
    """Run retrieval evaluation for all 8 subjects in parallel."""
    import json

    results = []
    skipped = []
    for subj, result in zip(
        ALL_SUBJECTS,
        compute_retrieval_subj.map(ALL_SUBJECTS, return_exceptions=True),
    ):
        if isinstance(result, Exception):
            print(f"  [{subj}] failed: {result}")
            skipped.append(subj)
        elif result:
            results.append(result)
        else:
            skipped.append(subj)

    out_dir = Path(MINDBRIDGE_ROOT) / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "retrieval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    volume.commit()

    summary = {
        "completed": results,
        "skipped": skipped,
        "out_file": str(out_file),
    }
    return summary


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
    script: str = "train_mindbridge_loss.py",
    epochs: int = 150,
    variant: str = "A",
    run_id: str = "",
    alpha: float = 1e4,
    ckpt_prefer: str = "final",
):
    """Run MindBridge on Modal.

    Examples:
      modal run modal_app.py --step download --all-subjects
      modal run modal_app.py --subj subj01 --step download
      modal run modal_app.py --subj subj01 --step ingest              # betas only
      modal run modal_app.py --subj subj01 --step ingest --compute-targets  # full
      modal run modal_app.py --step ingest-imagery --all-subjects   # imagery betas
      modal run modal_app.py --subj subj01 --step train
      modal run modal_app.py --subj subj01 --step train --script train_variant_a.py --epochs 3
      modal run modal_app.py --step train --all-subjects   # subj02-subj08 in parallel
      bash run_variant.sh a   # sanity test + all 8 subjects for variant A
    """
    if step == "download":
        if all_subjects:
            print(download_all.remote(sessions, stimuli, imagery))
        else:
            print(download_data.remote(subj, sessions, stimuli, imagery, False))
    elif step == "ingest":
        if all_subjects:
            print(ingest_all.remote(compute_targets))
        else:
            print(ingest.remote(subj, compute_targets))
    elif step == "ingest-imagery":
        if all_subjects:
            print(ingest_imagery_all.remote())
        else:
            print(ingest_imagery.remote(subj))
    elif step == "ingest-averaged":
        print("=== Averaged ingest (CPU): betas + vision targets ===")
        print(ingest_averaged_cpu.remote("all"))
        print("=== Averaged ingest (GPU): CLIP-text from COCO ===")
        print(ingest_averaged_text.remote())
    elif step == "ingest-averaged-cpu":
        print(ingest_averaged_cpu.remote("all" if all_subjects else subj))
    elif step == "ingest-averaged-text":
        print(ingest_averaged_text.remote())
    elif step == "train-4head":
        if all_subjects:
            args = [(s, "train_4head.py", epochs, run_id) for s in ALL_SUBJECTS]
            print(list(train.starmap(args)))
        else:
            print(train.remote(subj, "train_4head.py", epochs, run_id))
    elif step == "train-4head-contrastive":
        if all_subjects:
            args = [(s, "train_4head_contrastive.py", epochs, run_id) for s in ALL_SUBJECTS]
            print(list(train.starmap(args)))
        else:
            extra = [
                "--warmup-epochs", os.environ.get("MINDBRIDGE_WARMUP_EPOCHS", "15"),
                "--w-contrastive", os.environ.get("MINDBRIDGE_W_CONTRASTIVE", "2.5"),
            ]
            if os.environ.get("MINDBRIDGE_TEMPERATURE"):
                extra.extend(["--temperature", os.environ["MINDBRIDGE_TEMPERATURE"]])
            print(train.remote(subj, "train_4head_contrastive.py", epochs, run_id, extra))
    elif step == "train-4head-contrastive-sweep":
        base_rid = run_id or "20260601_4head_ctr_sweep"
        temps = [0.03, 0.05, 0.07, 0.1]
        print(f"=== τ sweep subj={subj} temps={temps} warmup=15 w_ctr=2.5 ===")
        sweep_args = []
        for t in temps:
            tag = str(t).replace(".", "")
            rid = f"{base_rid}_t{tag}"
            extra = [
                "--temperature", str(t),
                "--warmup-epochs", "15",
                "--w-contrastive", "2.5",
            ]
            sweep_args.append((subj, "train_4head_contrastive.py", epochs, rid, extra))
        results = list(train.starmap(sweep_args))
        print("Sweep results:", results)
    elif step == "train-4head-roi":
        baseline = os.environ.get(
            "MINDBRIDGE_BASELINE_RUN_ID", "20260601_4head_150ep_all8",
        )
        os.environ["MINDBRIDGE_BASELINE_RUN_ID"] = baseline
        extra = ["--baseline-run-id", baseline]
        if all_subjects:
            args = [
                (s, "train_4head_roi.py", epochs, run_id, extra) for s in ALL_SUBJECTS
            ]
            print(list(train.starmap(args)))
        else:
            print(train.remote(subj, "train_4head_roi.py", epochs, run_id, extra))
    elif step == "train":
        if all_subjects:
            print(train_all.remote())
        else:
            print(train.remote(subj, script, epochs, run_id))
    elif step == "reconstruct":
        print(reconstruct.remote(subj, n, vd))
    elif step == "reconstruct-imagery":
        print(reconstruct_imagery_small.remote(subj, 3, 0.6, run_id, ckpt_prefer))
    elif step == "reconstruct-perception":
        rid = run_id or "20260601_052537_varA_subj01_150ep"
        print(reconstruct_perception_small.remote(subj, 3, 0.6, rid, ckpt_prefer))
    elif step == "reconstruct-setb-best-clip":
        rid = run_id or "20260601_052537_varA_subj01_150ep"
        prefer = ckpt_prefer if ckpt_prefer != "final" else "best_clip"
        print(f"=== Set-B recon (imagery + perception) run={rid} ckpt={prefer} ===")
        h_imagery = reconstruct_imagery_small.spawn(subj, 3, 0.6, rid, prefer)
        h_perception = reconstruct_perception_small.spawn(subj, 3, 0.6, rid, prefer)
        print("Imagery:", h_imagery.get())
        print("Perception:", h_perception.get())
    elif step == "reconstruct-setb-4h-roi":
        prefer = ckpt_prefer if ckpt_prefer != "final" else "best_clip"
        rid_4h = run_id or "20260601_4head_150ep_all8"
        rid_roi = os.environ.get("MINDBRIDGE_ROI_RUN_ID", "20260601_4head_roi_subj01")
        print(f"=== Set-B imagery recon 4H vs ROI (best_clip) subj={subj} ===")
        h_4h = reconstruct_imagery_4head.spawn(subj, 3, 0.6, rid_4h, "4H", prefer)
        h_roi = reconstruct_imagery_4head.spawn(subj, 3, 0.6, rid_roi, "4head_roi", prefer)
        print("Global 4H:", h_4h.get())
        print("ROI 4-head:", h_roi.get())
    elif step == "retrain-subj01-imagery":
        from datetime import datetime, timezone

        rid = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_varA_subj01_150ep"
        print(f"=== Run ID: {rid} ===")
        print("Training Variant A subj01 (150 epochs)...")
        print(train.remote("subj01", "train_variant_a.py", epochs, rid))
        print("Reconstructing imagery...")
        print(reconstruct_imagery_small.remote("subj01", 3, 0.6, rid))
    elif step == "retrain-var-a-imagery":
        from datetime import datetime, timezone

        rid = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_varA_{epochs}ep"
        print(f"=== Run ID: {rid} ===")
        print("Training Variant A (150 epochs) for all 8 subjects in parallel...")
        train_args = [(s, "train_variant_a.py", epochs, rid) for s in ALL_SUBJECTS]
        results = list(train.starmap(train_args))
        print("Training results:", results)
        print("Reconstructing imagery for subj01...")
        print(reconstruct_imagery_small.remote("subj01", 3, 0.6, rid))
    elif step == "joint-ridge":
        if all_subjects:
            print(train_joint_ridge_all.remote(alpha, run_id))
        else:
            print(train_joint_ridge_subj.remote(subj, alpha, run_id))
    elif step == "check-ckpts":
        print(check_variant_ckpts.remote(variant))
    elif step == "check":
        print(check_nsd_volume.remote())
    elif step == "probe-roi":
        print(probe_nsd_roi.remote())
    elif step == "inspect-roi":
        print(inspect_roi_labels.remote(subj))
    elif step == "retrieval":
        if all_subjects or subj == "all":
            print(compute_retrieval_all.remote())
        else:
            print(compute_retrieval_subj.remote(subj))
    elif step == "eval-4head-retrieval":
        extra = ["--only-var-a"] if os.environ.get("EVAL_ONLY_VAR_A") == "1" else []
        print(train.remote(subj, "eval_4head_retrieval.py", 1, run_id, extra))
    elif step == "ingest-clip-hidden":
        print(ingest_clip_hidden.remote(subj if not all_subjects else "all"))
    elif step == "train-retrieval-frozen":
        extra = [
            "--target-mode", os.environ.get("RETRIEVAL_TARGET", "hidden257"),
            "--temperature", os.environ.get("MINDBRIDGE_TEMPERATURE", "0.05"),
            "--backbone-run-id", os.environ.get("BACKBONE_RUN_ID", "20260601_4head_150ep_all8"),
        ]
        print(train.remote(subj, "train_retrieval_frozen.py", epochs, run_id, extra))
    elif step == "train-retrieval-sweep":
        rid_base = run_id or "20260601_retrieval_sweep"
        modes = os.environ.get("RETRIEVAL_MODES", "hidden257,cls768").split(",")
        temps = [0.01, 0.03, 0.05, 0.07]
        args_list = []
        for mode in modes:
            for t in temps:
                tag = f"{mode}_t{str(t).replace('.', '')}"
                extra = [
                    "--target-mode", mode.strip(),
                    "--temperature", str(t),
                    "--backbone-run-id", "20260601_4head_150ep_all8",
                ]
                args_list.append((subj, "train_retrieval_frozen.py", epochs, f"{rid_base}_{tag}", extra))
        print(f"=== retrieval sweep {len(args_list)} jobs ===")
        print(list(train.starmap(args_list)))
    elif step == "eval-whitening":
        extra = ["--split", "both", "--variant", "A"]
        if os.environ.get("WHITEN_VARIANT"):
            extra = ["--split", os.environ.get("WHITEN_SPLIT", "both"), "--variant", os.environ["WHITEN_VARIANT"]]
        print(train.remote(subj, "eval_clip_whitening.py", 1, run_id, extra))
    elif step == "download-roi":
        print(download_roi_all.remote())
    elif step == "download-imagery-meta":
        print(download_imagery_meta.remote())
    else:
        print(run_pipeline.remote(subj))
