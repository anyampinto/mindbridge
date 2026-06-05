"""MindBridge pipeline on Modal: ingestion -> training -> reconstruction."""

import json
import os
import sys
from pathlib import Path

import modal

MINDBRIDGE_ROOT = "/mnt/mindbridge"
VOLUME_NAME = "mindbridge-data"
SCRIPTS_DIR = "/root/scripts"
BRAIN_DIFFUSER_DIR = "/root/brain-diffuser"

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
        "easydict",
        "pyyaml",
        "einops",
        "omegaconf",
        "lpips",
    )
    .add_local_dir("scripts", remote_path=SCRIPTS_DIR)
    .add_local_dir("models", remote_path="/root/models")
    .add_local_dir("vendor/brain-diffuser", remote_path=BRAIN_DIFFUSER_DIR)
)


def _configure_env(subj: str = "subj01") -> None:
    os.environ["MINDBRIDGE_ROOT"] = MINDBRIDGE_ROOT
    os.environ["NSD_SUBJ"] = subj
    os.environ["BRAIN_DIFFUSER_ROOT"] = BRAIN_DIFFUSER_DIR
    os.environ["TRANSFORMERS_CACHE"] = f"{MINDBRIDGE_ROOT}/hf_cache"
    if "/root/models" not in sys.path:
        sys.path.insert(0, "/root")
    os.environ["HF_HOME"] = f"{MINDBRIDGE_ROOT}/hf_cache"
    os.environ["TORCH_HOME"] = f"{MINDBRIDGE_ROOT}/torch_cache"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token, add_to_git_credential=False)
        except Exception as exc:
            print(f"  WARNING: HF login failed: {exc}")

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
def ingest_imagery_sets(subj: str = "subj01", sets: str = "C"):
    """Ingest selected imagery set averages only; merge into existing betas_avg / meta."""
    volume.reload()
    _configure_env(subj)
    from ingest_imagery_betas import run_ingestion

    summary = run_ingestion(subj=subj, root=Path(MINDBRIDGE_ROOT), sets=sets)
    volume.commit()
    return summary


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
        sys.argv = [script, "--subj", subj, "--root", MINDBRIDGE_ROOT]
        if not (script.startswith("eval_") or script.startswith("compute_")):
            sys.argv.extend(["--epochs", str(epochs)])
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
    variant: str = "A",
    clip_source: str = "regression",
    all_set_b: bool = False,
    eval_clip_2wc: bool = False,
):
    """Reconstruct Set-B imagery stimuli."""
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
        "--variant", variant,
        "--clip-source", clip_source,
    ]
    if all_set_b:
        argv.append("--all-set-b")
    if eval_clip_2wc:
        argv.append("--eval-clip-2wc")
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
    variant: str = "A",
    clip_source: str = "regression",
):
    """Reconstruct same Set-B NSD images from perception betas."""
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
        "--variant", variant,
        "--clip-source", clip_source,
    ]
    recon_main()
    volume.commit()
    return f"Perception reconstruction saved for {subj} (run={run_id})"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 6, volumes={MINDBRIDGE_ROOT: volume})
def reconstruct(
    subj: str = "subj01",
    mode: str = "both",
    run_id: str = "",
    n_perception: int = 3,
    vd_strength: float = 0.6,
    vd_steps: int = 20,
    decoder: str = "vd",
    variant: str = "4H_CTR",
    ckpt_prefer: str = "best_retrieval",
    n_imagery: int = 3,
    prior_strength: float = 0.25,
    kandinsky_strength: float = 0.3,
):
    """Reconstruction: perception val + imagery Set B grids + CLIP 2WC."""
    _configure_env(subj)
    from reconstruct import main as reconstruct_main

    argv = [
        "reconstruct.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--decoder", decoder,
        "--mode", mode,
        "--n-perception", str(n_perception),
        "--n-imagery", str(n_imagery),
    ]
    if decoder == "kandinsky":
        rid = run_id or "20260602_034428_train_contrastive_retrieval_subj01_150ep"
        argv.extend([
            "--run-id", rid,
            "--variant", variant,
            "--ckpt-prefer", ckpt_prefer,
            "--clip-source", "regression",
            "--prior-strength", str(prior_strength),
            "--kandinsky-strength", str(kandinsky_strength),
        ])
    else:
        argv.extend([
            "--vd-strength", str(vd_strength),
            "--vd-steps", str(vd_steps),
        ])
        if run_id:
            argv.extend(["--run-id", run_id])
    sys.argv = argv
    reconstruct_main()
    volume.commit()
    return f"{decoder} reconstruction ({mode}) complete for {subj}"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 12, volumes={MINDBRIDGE_ROOT: volume})
def vdvae_encode(subj: str = "subj01", all_subjects: bool = False, skip_global: bool = False):
    """Encode NSD images with brain-diffuser VDVAE → vdvae_latents_subj{XX}.npy."""
    volume.reload()
    _configure_env(subj)
    import sys
    from vdvae_encoder_inference import main as enc_main

    argv = ["vdvae_encoder_inference.py", "--root", MINDBRIDGE_ROOT]
    if all_subjects:
        argv.append("--all")
    else:
        argv.extend(["--subj", subj])
    if skip_global:
        argv.append("--skip-global")
    sys.argv = argv
    enc_main()
    volume.commit()
    return "VDVAE encode complete"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 10, volumes={MINDBRIDGE_ROOT: volume})
def vdvae_ref_dims():
    """Write layer_flat_dims to vdvae/ref_stats.npz (GPU, ~1 min)."""
    volume.reload()
    _configure_env("subj01")
    from vdvae_write_ref_dims import write_ref_dims

    write_ref_dims(Path(MINDBRIDGE_ROOT))
    volume.commit()
    return "ref_stats layer_flat_dims written"


@app.function(
    image=image,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={MINDBRIDGE_ROOT: volume},
)
def vdvae_regression_subj(subj: str = "subj01", run_id: str = "", alpha: float = 50000.0):
    """Unaveraged trial ridge → 31-layer VDVAE latents (CPU, 64GB RAM)."""
    volume.reload()
    _configure_env(subj)
    import sys
    from vdvae_regression import main as reg_main

    argv = [
        "vdvae_regression.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--data", "trials",
        "--alpha", str(alpha),
    ]
    if run_id:
        argv.extend(["--run-id", run_id])
    sys.argv = argv
    reg_main()
    volume.commit()
    return f"VDVAE trial ridge complete for {subj}"


@app.function(
    image=image,
    memory=65536,
    timeout=60 * 60 * 12,
    volumes={MINDBRIDGE_ROOT: volume},
)
def vdvae_regression_all(run_id: str = "", alpha: float = 50000.0):
    """Trial-level ridge for all 8 subjects in parallel."""
    args = [(s, run_id, alpha) for s in ALL_SUBJECTS]
    return list(vdvae_regression_subj.starmap(args))


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 6, volumes={MINDBRIDGE_ROOT: volume})
def reconstruct_vdvae_dual(
    subj: str = "subj01",
    run_id: str = "",
    variant: str = "4H_CTR2",
    ckpt_prefer: str = "best_retrieval",
    n_perception: int = 3,
    vd_strength: float = 0.75,
    vd_steps: int = 50,
    guidance_scale: float = 7.5,
    perception_only: bool = False,
    imagery_only: bool = False,
    beta_adapter_run_id: str = "",
    stage2_source: str = "projector",
    prior_run_id: str = "",
    prior_temperature: float = 1.0,
    eval_recon_kneeland: bool = False,
    out_subdir: str = "",
    text_source: str = "",
    vd_vision_mix: float = 0.6,
    text_to_image_strength: float = -1.0,
    brain_token_weight: float = 0.85,
    prior_seed: int = -1,
    vd_seed: int = -1,
    imagery_sets: str = "set_b",
    prior_blend_weight: float = 0.85,
    kneeland_b_vd_strength: float = -1.0,
    kneeland_a_text_source: str = "",
    kneeland_b_text_source: str = "",
    stage1_source: str = "vdvae_ridge",
    adapter_train_subj: str = "",
    use_pooled_prior: bool = False,
    perception_setb_only: bool = False,
):
    """VDVAE Stage 1 + dual projector VD Stage 2 (Set B + perception val grids)."""
    volume.reload()
    _configure_env(subj)
    if adapter_train_subj:
        os.environ["ADAPTER_TRAIN_SUBJ"] = adapter_train_subj
    if use_pooled_prior:
        os.environ["USE_POOLED_PRIOR"] = "1"
    import sys
    from reconstruct_vdvae_dual import main as vdvae_recon_main

    rid = run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
    sys.argv = [
        "reconstruct_vdvae_dual.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--run-id", rid,
        "--variant", variant,
        "--ckpt-prefer", ckpt_prefer,
        "--n-perception", str(n_perception),
        "--vd-strength", str(vd_strength),
        "--vd-steps", str(vd_steps),
        "--guidance-scale", str(guidance_scale),
        "--stage2-source", stage2_source,
        "--stage1-source", stage1_source,
    ]
    if perception_only:
        sys.argv.append("--perception-only")
    if perception_setb_only:
        sys.argv.append("--perception-setb-only")
    if imagery_only:
        sys.argv.append("--imagery-only")
    if beta_adapter_run_id:
        sys.argv.extend(["--beta-adapter-run-id", beta_adapter_run_id])
    if prior_run_id:
        sys.argv.extend(["--prior-run-id", prior_run_id])
    if prior_temperature != 1.0:
        sys.argv.extend(["--prior-temperature", str(prior_temperature)])
    if out_subdir:
        sys.argv.extend(["--out-subdir", out_subdir])
    elif stage2_source == "prior":
        sys.argv.extend(["--out-subdir", "prior_adapted"])
    if eval_recon_kneeland:
        sys.argv.append("--eval-recon-kneeland")
    if text_source:
        sys.argv.extend(["--text-source", text_source])
    if text_source and vd_vision_mix != 0.6:
        sys.argv.extend(["--vd-vision-mix", str(vd_vision_mix)])
    if text_to_image_strength >= 0:
        sys.argv.extend(["--text-to-image-strength", str(text_to_image_strength)])
    if text_source == "mlp_text_hf_dual" and brain_token_weight != 0.85:
        sys.argv.extend(["--brain-token-weight", str(brain_token_weight)])
    if prior_seed >= 0:
        sys.argv.extend(["--prior-seed", str(prior_seed)])
    if vd_seed >= 0:
        sys.argv.extend(["--vd-seed", str(vd_seed)])
    if imagery_sets and imagery_sets != "set_b":
        sys.argv.extend(["--imagery-sets", imagery_sets])
    if stage2_source == "prior_blend":
        sys.argv.extend(["--prior-blend-weight", str(prior_blend_weight)])
    if kneeland_b_vd_strength >= 0:
        sys.argv.extend(["--kneeland-b-vd-strength", str(kneeland_b_vd_strength)])
    if kneeland_a_text_source:
        sys.argv.extend(["--kneeland-a-text-source", kneeland_a_text_source])
    if kneeland_b_text_source:
        sys.argv.extend(["--kneeland-b-text-source", kneeland_b_text_source])
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        vdvae_recon_main()
    out = buf.getvalue()
    volume.commit()
    metric_lines: list[str] = []
    for line in out.splitlines():
        if line.startswith((
            "KNEELAND_2WC_RESULT=", "KNEELAND_2WC_AVG_AB=", "KNEELAND_2WC_PAPER_AVG_AB=",
            "KNEELAND_2WC_GT_DIST_AVG_AB=", "KNEELAND_2WC_SET_A=", "KNEELAND_2WC_SET_B=",
            "KNEELAND_2WC_SET_A_PAPER=", "KNEELAND_2WC_SET_B_PAPER=",
            "CLIP_GT_COSIM=", "QUALITY_SCORE=",
        )):
            metric_lines.append(line.strip())
    if metric_lines:
        return "\n".join(metric_lines)
    if imagery_only:
        label = "imagery Set-B"
    elif perception_only:
        label = "perception val"
    else:
        label = "full"
    return f"VDVAE dual recon ({label}) complete for {subj}"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 30, volumes={MINDBRIDGE_ROOT: volume})
def reeval_recon_grid(
    subj: str = "subj01",
    grid_rel: str = "reconstructions/subj01/prior_adapted_vd03_t15/imagery_setB_grid.png",
    stage2_dir_rel: str = "",
    n_pairs: int = 1000,
    label: str = "",
    expected_k2wc: float = -1.0,
):
    """Re-run Kneeland 2WC on saved Stage-2 imagery outputs (eval sanity check)."""
    volume.reload()
    _configure_env(subj)
    import sys
    from reeval_recon_grid import main as reeval_main

    sys.argv = [
        "reeval_recon_grid.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--n-pairs", str(n_pairs),
    ]
    if stage2_dir_rel:
        sys.argv.extend(["--stage2-dir", f"{MINDBRIDGE_ROOT}/{stage2_dir_rel.lstrip('/')}"])
    else:
        sys.argv.extend(["--grid", f"{MINDBRIDGE_ROOT}/{grid_rel.lstrip('/')}"])
    if label:
        sys.argv.extend(["--label", label])
    if expected_k2wc >= 0:
        sys.argv.extend(["--expected-k2wc", str(expected_k2wc)])
    reeval_main()
    volume.commit()
    return f"Re-eval complete: {stage2_dir_rel or grid_rel}"


@app.function(image=image, gpu=gpu_fn, timeout=60 * 30, volumes={MINDBRIDGE_ROOT: volume})
def eval_kneeland_paper_runs(
    subj: str = "subj01",
    run_dirs: str = "kn_atxt_a_txt005_b_vd03_s29,kn_ab_kn_vd03_s29,push60_prior_s29",
    n_pairs: int = 1000,
    seed: int = 42,
):
    """Re-score saved Kneeland A+B runs: legacy GT-distractor vs paper recon-distractor 2WC."""
    volume.reload()
    _configure_env(subj)
    import json
    from pathlib import Path

    from eval_recon_kneeland import eval_kneeland_ab_from_run_dir
    from paths import get_root

    root = Path(MINDBRIDGE_ROOT)
    device = __import__("torch").device("cuda")
    rows_out = []
    print("=== Kneeland paper vs legacy 2WC (Appendix A.2 recon distractors) ===\n")
    for name in run_dirs.split(","):
        name = name.strip()
        if not name:
            continue
        out_dir = root / "reconstructions" / subj / name
        if not out_dir.exists():
            print(f"  SKIP {name}: missing {out_dir}")
            continue
        try:
            res = eval_kneeland_ab_from_run_dir(
                out_dir, subj, root, n_pairs=n_pairs, device=device, seed=seed,
            )
        except Exception as exc:
            print(f"  FAIL {name}: {exc}")
            continue
        row = {
            "run": name,
            "avg_ab_gt_distractor": res["avg_ab_gt_distractor"],
            "avg_ab_paper": res["avg_ab_paper"],
            "set_a_paper": res["set_a"]["kneeland_2wc_paper"]["kneeland_2wc"],
            "set_b_paper": res["set_b"]["kneeland_2wc_paper"]["kneeland_2wc"],
        }
        rows_out.append(row)
        print(
            f"  {name}:  legacy A+B={100 * res['avg_ab_gt_distractor']:.1f}%  "
            f"paper A+B={100 * res['avg_ab_paper']:.1f}%  "
            f"(A paper {100 * row['set_a_paper']:.1f}%  B paper {100 * row['set_b_paper']:.1f}%)"
        )
    out_json = root / "results" / f"kneeland_paper_comparison_{subj}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"subj": subj, "n_pairs": n_pairs, "seed": seed, "runs": rows_out}, indent=2))
    print(f"\n  Saved {out_json}")
    volume.commit()
    best = max(rows_out, key=lambda r: r["avg_ab_paper"]) if rows_out else None
    if best:
        print(
            f"KNEELAND_2WC_PAPER_AVG_AB={best['avg_ab_paper']:.6f}  "
            f"(best run={best['run']})"
        )
    return json.dumps(rows_out, indent=2)


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 8, volumes={MINDBRIDGE_ROOT: volume})
def kneeland_multisample_atxt(
    subj: str = "subj01",
    seeds: str = "20,21,22,23,24,25,26,27,28,29",
    dual_run_id: str = "",
    beta_adapter_run_id: str = "",
    prior_run_id: str = "",
    text_strength: float = 0.05,
    prior_temperature: float = 1.5,
    vd_b: float = 0.3,
    out_base: str = "kn_atxt_ms",
    n_pairs: int = 1000,
):
    """10-sample Kneeland: A=MLP text, B=prior; full Set-B gallery (incl. T); paper multisample 2WC."""
    volume.reload()
    _configure_env(subj)
    import json
    from pathlib import Path

    from reconstruct_vdvae_dual import run_kneeland_multisample_sweep

    rid = dual_run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
    beta = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
    seed_list = [int(x) for x in seeds.split(",")]
    print(
        f"=== Kneeland multisample ({len(seed_list)} seeds) A=text@{text_strength} B=prior "
        f"VD={vd_b} τ={prior_temperature} ==="
    )
    res = run_kneeland_multisample_sweep(
        subj,
        Path(MINDBRIDGE_ROOT),
        seeds=seed_list,
        run_id=rid,
        beta_adapter_run_id=beta,
        prior_run_id=prior_run_id,
        kneeland_a_text_source="mlp_text_hf_dual",
        text_to_image_strength=text_strength,
        prior_temperature=prior_temperature,
        kneeland_b_vd_strength=vd_b,
        out_subdir_base=out_base,
        n_pairs=n_pairs,
    )
    volume.commit()
    return json.dumps(res, indent=2)


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 2, volumes={MINDBRIDGE_ROOT: volume})
def stage1_diagnose(
    subj: str = "subj01",
    clip_adapter_run_id: str = "",
    vdvae_adapter_run_id: str = "",
    full_layers: bool = False,
    out_subdir: str = "stage1_diagnose",
):
    """Set-B Stage-1 grid: GT ceiling vs imagery / perception / VDVAE-adapted betas."""
    volume.reload()
    _configure_env(subj)
    import json
    from pathlib import Path

    from stage1_diagnose import run_stage1_diagnose

    clip = clip_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
    vdvae = vdvae_adapter_run_id or "20260604_060546_imagery_vdvae_beta_adapter_subj01"
    print(f"=== Stage-1 diagnose Set-B subj={subj} full_layers={full_layers} ===")
    res = run_stage1_diagnose(
        subj,
        Path(MINDBRIDGE_ROOT),
        clip_adapter_run_id=clip,
        vdvae_adapter_run_id=vdvae,
        hybrid_layers=not full_layers,
        full_layers=full_layers,
        out_subdir=out_subdir,
    )
    volume.commit()
    return json.dumps(res, indent=2)


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 3, volumes={MINDBRIDGE_ROOT: volume})
def semantics_compare(
    subj: str = "subj01",
    dual_run_id: str = "",
    beta_adapter_run_id: str = "",
    prior_run_id: str = "",
    prior_temperature: float = 1.5,
    seed: int = 29,
    out_subdir: str = "semantics_compare_s29",
):
    """GT + VDVAE S1 + unCLIP + 3 VD configs on Set-B W/K/B/C/D (CLIP→GT per row)."""
    volume.reload()
    _configure_env(subj)
    import json
    from pathlib import Path

    from semantics_compare import run_semantics_compare

    rid = dual_run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
    beta = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
    print(f"=== Semantics compare Set-B seed={seed} τ={prior_temperature} ===")
    res = run_semantics_compare(
        subj,
        Path(MINDBRIDGE_ROOT),
        run_id=rid,
        beta_adapter_run_id=beta,
        prior_run_id=prior_run_id or None,
        prior_temperature=prior_temperature,
        seed=seed,
        out_subdir=out_subdir,
    )
    volume.commit()
    return json.dumps(res, indent=2)


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 8, volumes={MINDBRIDGE_ROOT: volume})
def recon_pipeline_subj01_modal(
    subj: str = "subj01",
    dual_run_id: str = "",
    out_subdir: str = "kn_atxt_a_txt005_b_vd03_s29",
    seed: int = 29,
):
    """
    Production Kneeland A+B recon (~65% recipe). Grids only — no retraining.
    Same as kneeland-calibrated-recon for subj01.
    """
    volume.reload()
    _configure_env(subj)
    from recon_pipeline_subj01 import PRODUCTION_ADAPTER_RUN, PRODUCTION_DUAL_RUN, run_pipeline

    summary = run_pipeline(
        subj,
        MINDBRIDGE_ROOT,
        dual_run_id=dual_run_id or os.environ.get("DUAL_RUN_ID", PRODUCTION_DUAL_RUN),
        beta_adapter_run_id=os.environ.get("BETA_ADAPTER_RUN", PRODUCTION_ADAPTER_RUN),
        out_subdir=os.environ.get("KNEELAND_OUT", os.environ.get("PIPELINE_OUT", out_subdir)),
        seed=int(os.environ.get("PRIOR_SEED", str(seed))),
        vd_strength=float(os.environ.get("VD_STRENGTH", "0.3")),
        prior_temperature=float(os.environ.get("PRIOR_TEMPERATURE", "1.5")),
        hf_text_strength=float(os.environ.get("HF_TEXT_STRENGTH", "0.05")),
        stage1_source=os.environ.get("STAGE1_SOURCE", "vdvae_ridge"),
        n_perception=int(os.environ.get("N_PERCEPTION", "3")),
    )
    volume.commit()
    return json.dumps(summary, indent=2, default=str)


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 4, volumes={MINDBRIDGE_ROOT: volume})
def train_diffusion_prior_subj(
    subj: str = "subj01",
    run_id: str = "",
    dual_run_id: str = "",
    epochs: int = 150,
    batch_size: int = 128,
):
    """Train 1024-d projector → 768-d CLIP diffusion prior on perception trials."""
    volume.reload()
    _configure_env(subj)
    if run_id:
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    from train_diffusion_prior import run_training

    dual_rid = dual_run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
    result = run_training(
        subj,
        Path(MINDBRIDGE_ROOT),
        dual_run_id=dual_rid,
        run_id=run_id or None,
        epochs=epochs,
        batch_size=batch_size,
    )
    volume.commit()
    return result


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 4, volumes={MINDBRIDGE_ROOT: volume})
def reconstruct_imagery_unclip(
    subj: str = "subj01",
    variant: str = "joint_ridge",
    run_id: str = "",
    subset: str = "set_b",
    clip_source: str = "projector",
    ckpt_prefer: str = "best_retrieval",
    steps: int = 50,
    guidance_scale: float = 7.5,
    beta_adapter_run_id: str = "",
    eval_recon_kneeland: bool = False,
    eval_embedding_kneeland: bool = False,
):
    """Imagery SD unCLIP from joint ridge or dual projector CLIP (no VDVAE)."""
    volume.reload()
    _configure_env(subj)
    import sys
    from reconstruct_imagery_unclip import main as unclip_main

    sys.argv = [
        "reconstruct_imagery_unclip.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--variant", variant,
        "--subset", subset,
        "--clip-source", clip_source,
        "--ckpt-prefer", ckpt_prefer,
        "--steps", str(steps),
        "--guidance-scale", str(guidance_scale),
    ]
    if run_id:
        sys.argv.extend(["--run-id", run_id])
    if beta_adapter_run_id:
        sys.argv.extend(["--beta-adapter-run-id", beta_adapter_run_id])
    if eval_recon_kneeland:
        sys.argv.append("--eval-recon-kneeland")
    if eval_embedding_kneeland:
        sys.argv.append("--eval-embedding-kneeland")
    unclip_main()
    volume.commit()
    return f"Imagery unCLIP ({variant}, {subset}) complete for {subj}"


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
    timeout=60 * 30,
    volumes={MINDBRIDGE_ROOT: volume},
)
def roi_corr_audit_subj(train_subj: str = "subj01"):
    """Build nsdgeneral grid correspondence for cross-subject beta adapter."""
    volume.reload()
    _configure_env(train_subj)
    import sys

    sys.argv = [
        "roi_correspondence.py", "--audit-all", "--train-subj", train_subj,
        "--root", MINDBRIDGE_ROOT,
    ]
    from roi_correspondence import main as roi_corr_main

    roi_corr_main()
    volume.commit()
    return f"roi_corr_audit done for train={train_subj}"


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


@app.function(image=image, gpu=gpu_fn, timeout=60 * 30, volumes={MINDBRIDGE_ROOT: volume})
def eval_beta_alignment_subj(
    subj: str = "subj01",
    adapter_run_id: str = "",
    dual_run_id: str = "",
):
    """Report img→perc vs adapted→perc cosine on paired Kneeland stimuli."""
    volume.reload()
    _configure_env(subj)
    import json
    import sys
    from pathlib import Path

    from eval_beta_alignment import main as align_main

    rid = adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
    dual = dual_run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
    sys.argv = [
        "eval_beta_alignment.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--adapter-run-id", rid,
        "--dual-run-id", dual,
    ]
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        align_main()
    out = buf.getvalue()
    print(out)
    volume.commit()
    return out


@app.function(
    image=image,
    gpu=gpu_fn,
    timeout=60 * 180,
    volumes={MINDBRIDGE_ROOT: volume},
)
def train_imagery_beta_adapter_subj(
    subj: str = "subj01",
    run_id: str = "",
    epochs: int = 200,
    align_space: str = "mlp",
    dual_run_id: str = "",
    beta_weight: float = 0.35,
    init_adapter_run_id: str = "",
    exclude_sets_from_fit: str = "",
    skip_loo: bool = False,
):
    """Contrastive imagery→perception beta adapter."""
    volume.reload()
    _configure_env(subj)
    if run_id:
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    from train_imagery_beta_adapter import run_training

    excl = frozenset(s.strip() for s in exclude_sets_from_fit.split(",") if s.strip())
    result = run_training(
        subj,
        Path(MINDBRIDGE_ROOT),
        run_id=run_id or None,
        align_space=align_space,
        dual_run_id=dual_run_id or "20260602_053625_train_dual_contrastive_subj01_150ep",
        epochs=epochs,
        beta_weight=beta_weight,
        init_adapter_run_id=init_adapter_run_id or None,
        exclude_sets_from_fit=excl or None,
        skip_loo=skip_loo,
    )
    volume.commit()
    return result


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 12, volumes={MINDBRIDGE_ROOT: volume})
def honest_kneeland_pipeline_subj(
    subj: str = "subj01",
    run_id: str = "",
    epochs: int = 200,
    skip_loo: bool = False,
    prior_seed: int = 29,
    vd_strength: float = 0.3,
    text_strength: float = 0.05,
    out_subdir: str = "",
):
    """Honest adapter (fit excludes A/B) + Kneeland A+B recon in one GPU job."""
    volume.reload()
    _configure_env(subj)
    import io
    import json
    import sys
    from contextlib import redirect_stdout
    from pathlib import Path

    from train_imagery_beta_adapter import run_training

    dual_rid = "20260602_053625_train_dual_contrastive_subj01_150ep"
    osd = out_subdir or f"kn_atxt_honest_s{prior_seed}"
    s1_src = os.environ.get("STAGE1_SOURCE", "vdvae_ridge")
    print(
        f"=== Honest pipeline subj={subj} LOO={not skip_loo} fit=Set-C-only "
        f"stage1={s1_src} → {osd} ==="
    )

    train_res = run_training(
        subj,
        Path(MINDBRIDGE_ROOT),
        run_id=run_id or None,
        align_space="mlp",
        dual_run_id=dual_rid,
        exclude_sets_from_fit=frozenset({"A", "B"}),
        fit_only_sets=frozenset({"C"}),
        skip_loo=skip_loo,
        epochs=epochs,
    )
    honest_rid = train_res["run_id"]
    print(f"ADAPTER_RUN_ID={honest_rid}")

    from reconstruct_vdvae_dual import main as vdvae_recon_main

    sys.argv = [
        "reconstruct_vdvae_dual.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--run-id", dual_rid,
        "--variant", "4H_CTR2",
        "--ckpt-prefer", "best_retrieval",
        "--n-perception", "3",
        "--vd-strength", str(vd_strength),
        "--vd-steps", "50",
        "--guidance-scale", "7.5",
        "--stage2-source", "prior",
        "--prior-temperature", "1.5",
        "--beta-adapter-run-id", honest_rid,
        "--eval-recon-kneeland",
        "--out-subdir", osd,
        "--text-source", "",
        "--text-to-image-strength", str(text_strength),
        "--brain-token-weight", "0.85",
        "--prior-seed", str(prior_seed),
        "--vd-seed", str(prior_seed),
        "--imagery-sets", "kneeland",
        "--prior-blend-weight", "0.85",
        "--kneeland-b-vd-strength", str(vd_strength),
        "--kneeland-a-text-source", "mlp_text_hf_dual",
        "--kneeland-b-text-source", "",
        "--stage1-source", s1_src,
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        vdvae_recon_main()
    recon_out = buf.getvalue()
    print(recon_out)
    volume.commit()
    return json.dumps({"train": train_res, "adapter_run_id": honest_rid, "out_subdir": osd}, indent=2)


@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 24, volumes={MINDBRIDGE_ROOT: volume})
def kneeland_production_pipeline_subj(
    subj: str = "subj05",
    dual_run_id: str = "",
    epochs: int = 150,
    adapter_epochs: int = 200,
    prior_epochs: int = 150,
    skip_loo: bool = True,
    prior_seed: int = 29,
    out_subdir: str = "",
):
    """
    subj05 (or any): 4H_CTR2 dual train → VDVAE S1 → imagery adapter → diffusion prior
    → calibrated Kneeland A+B recon (same recipe as subj01 ~65% run).
    """
    volume.reload()
    _configure_env(subj)
    import io
    import json
    import sys
    from contextlib import redirect_stdout
    from pathlib import Path

    from paths import averaged_meta_dir, make_run_id

    root = Path(MINDBRIDGE_ROOT)
    dual_rid = dual_run_id or make_run_id(f"train_dual_contrastive_{subj}_{epochs}ep")
    osd = out_subdir or f"kn_atxt_{subj}_s{prior_seed}"
    seed = prior_seed
    tau = 1.5
    vd_b = 0.3
    txt = 0.05

    print(f"=== Kneeland production pipeline {subj} dual={dual_rid} → {osd} ===")

    avg_dir = averaged_meta_dir(root)
    perc_betas = avg_dir / f"betas_avg_perception_{subj}.npy"
    if not perc_betas.exists():
        raise FileNotFoundError(
            f"{perc_betas} missing — run: modal run modal_app.py --subj {subj} "
            "--step ingest-averaged-cpu (and ingest-averaged-text if needed)"
        )
    img_meta = root / "nsd_meta" / f"imagery_trial_meta_{subj}.json"
    img_betas = root / "nsd_meta" / f"betas_avg_imagery_{subj}.npy"
    if not img_meta.exists() or not img_betas.exists():
        raise FileNotFoundError(
            f"Imagery ingest missing for {subj} — run: modal run modal_app.py "
            f"--subj {subj} --step ingest-imagery"
        )

    hidden_path = avg_dir / f"targets_clip_hidden_{subj}.npy"
    if not hidden_path.exists():
        print(f"=== ingest_clip_hidden {subj} ===")
        sys.argv = [
            "ingest_clip_hidden.py", "--subj", subj, "--root", MINDBRIDGE_ROOT,
            "--batch-size", "16", "--device", "cuda",
        ]
        from ingest_clip_hidden import main as ingest_hidden_main
        ingest_hidden_main()
        volume.commit()

    os.environ["MINDBRIDGE_RUN_ID"] = dual_rid
    print(f"=== train_dual_contrastive 4H_CTR2 {subj} {epochs}ep run={dual_rid} ===")
    from train_dual_contrastive import run_training as run_dual_training
    dual_res = run_dual_training(subj=subj, root=root, epochs=epochs)
    volume.commit()

    print(f"=== VDVAE encode + trial ridge {subj} ===")
    sys.argv = ["vdvae_encoder_inference.py", "--root", MINDBRIDGE_ROOT, "--subj", subj]
    from vdvae_encoder_inference import main as vdvae_enc_main
    vdvae_enc_main()
    from vdvae_write_ref_dims import write_ref_dims
    write_ref_dims(root)
    sys.argv = [
        "vdvae_regression.py", "--subj", subj, "--root", MINDBRIDGE_ROOT,
        "--data", "trials", "--alpha", "50000",
    ]
    from vdvae_regression import main as vdvae_reg_main
    vdvae_reg_main()
    volume.commit()

    print(f"=== Imagery beta adapter {subj} (dual={dual_rid}) ===")
    from train_imagery_beta_adapter import run_training as run_adapter_training
    adapter_res = run_adapter_training(
        subj,
        root,
        align_space="mlp",
        dual_run_id=dual_rid,
        epochs=adapter_epochs,
        skip_loo=skip_loo,
    )
    adapter_rid = adapter_res["run_id"]
    volume.commit()

    prior_rid = make_run_id(f"diffusion_prior_{subj}")
    os.environ["MINDBRIDGE_RUN_ID"] = prior_rid
    print(f"=== Diffusion prior {subj} (dual={dual_rid}) run={prior_rid} ===")
    from train_diffusion_prior import run_training as run_prior_training
    prior_res = run_prior_training(
        subj, root, dual_run_id=dual_rid, run_id=prior_rid, epochs=prior_epochs,
    )
    volume.commit()

    print(f"=== Kneeland calibrated recon {subj} adapter={adapter_rid} ===")
    from reconstruct_vdvae_dual import main as vdvae_recon_main
    sys.argv = [
        "reconstruct_vdvae_dual.py",
        "--subj", subj,
        "--root", MINDBRIDGE_ROOT,
        "--run-id", dual_rid,
        "--variant", "4H_CTR2",
        "--ckpt-prefer", "best_retrieval",
        "--n-perception", "3",
        "--vd-strength", str(vd_b),
        "--vd-steps", "50",
        "--guidance-scale", "7.5",
        "--stage2-source", "prior",
        "--prior-temperature", str(tau),
        "--beta-adapter-run-id", adapter_rid,
        "--eval-recon-kneeland",
        "--out-subdir", osd,
        "--text-source", "",
        "--text-to-image-strength", str(txt),
        "--brain-token-weight", "0.85",
        "--prior-seed", str(seed),
        "--vd-seed", str(seed),
        "--imagery-sets", "kneeland",
        "--prior-blend-weight", "0.85",
        "--kneeland-b-vd-strength", str(vd_b),
        "--kneeland-a-text-source", "mlp_text_hf_dual",
        "--kneeland-b-text-source", "",
        "--stage1-source", "vdvae_ridge",
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        vdvae_recon_main()
    print(buf.getvalue())
    volume.commit()

    summary = {
        "subj": subj,
        "dual_run_id": dual_rid,
        "dual_train": dual_res,
        "adapter_run_id": adapter_rid,
        "prior_run_id": prior_res.get("run_id", prior_rid),
        "out_subdir": osd,
        "recon_dir": str(root / "reconstructions" / subj / osd),
    }
    print(json.dumps(summary, indent=2))
    return json.dumps(summary, indent=2)


@app.function(
    image=image,
    gpu=gpu_fn,
    timeout=60 * 60 * 8,
    volumes={MINDBRIDGE_ROOT: volume},
)
def train_imagery_vdvae_beta_adapter_subj(
    subj: str = "subj01",
    run_id: str = "",
    clip_adapter_run_id: str = "20260603_230807_imagery_beta_adapter_subj01",
    combined: bool = False,
    eval_recon: bool = False,
    epochs: int = 20,
):
    """LOO beta adapter for VDVAE early layers (init from CLIP adapter)."""
    volume.reload()
    _configure_env(subj)
    if run_id:
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
    from train_imagery_vdvae_beta_adapter import run_training

    result = run_training(
        subj,
        Path(MINDBRIDGE_ROOT),
        run_id=run_id or None,
        clip_adapter_run_id=clip_adapter_run_id,
        combined=combined,
        eval_recon=eval_recon,
        epochs=epochs,
    )
    volume.commit()
    return result


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

@app.function(
    image=image,
    timeout=60 * 5,
    volumes={MINDBRIDGE_ROOT: volume},
)
def inspect_ckpt(path: str):
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    print("\n=== FILE ===")
    print(path)

    print("\n=== TOP LEVEL KEYS ===")
    print(list(ckpt.keys()))

    sd = ckpt.get("model_state", ckpt)

    print("\n=== FIRST 30 STATE_DICT KEYS ===")
    for k in list(sd.keys())[:30]:
        print(k)
    
    print("VARIANT:", ckpt.get("variant"))
    print("EPOCH:", ckpt.get("epoch"))
    print("CONTRASTIVE TARGET:", ckpt.get("contrastive_target"))

    return "done"

@app.function(image=image, gpu=gpu_fn, timeout=60 * 60 * 2, volumes={MINDBRIDGE_ROOT: volume})
def eval_whitening(subj: str, run_id: str, variant: str = "4H_CTR2"):
    _configure_env(subj)

    import sys
    sys.argv = [
        "eval_clip_whitening.py",
        "--subj", subj,
        "--run-id", run_id,
        "--variant", variant,
        "--root", MINDBRIDGE_ROOT,
    ]

    import runpy
    runpy.run_path(f"{SCRIPTS_DIR}/eval_clip_whitening.py", run_name="__main__")

    return "whitening complete"

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
    clip_source: str = "regression",
    imagery_subset: str = "set_b",
    eval_head: str = "both",
    imagery_sets: str = "C",
    set: str = "AB",
    head: str = "image_projector",
    beta_adapter_run_id: str = "",
    source: str = "embedding",
    vd_strength: float = 0.75,
    prior_temperature: float = 1.0,
    text_source: str = "",
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
    elif step == "ingest-imagery-setc":
        print(f"=== Ingest Set C imagery only (merge) subj={subj} ===")
        print(ingest_imagery_sets.remote(subj, "C"))
    elif step == "ingest-imagery-sets":
        print(f"=== Ingest imagery sets {imagery_sets} (merge) subj={subj} ===")
        print(ingest_imagery_sets.remote(subj, imagery_sets))
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
    elif step in ("reconstruct", "reconstruct-dual"):
        rid = run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
        recon_mode = os.environ.get("MINDBRIDGE_RECON_MODE", "both")
        print(f"=== Dual recon + CLIP 2WC subj={subj} mode={recon_mode} run={rid} ===")
        print(reconstruct.remote(subj, recon_mode, rid, 3, 0.6, 20, "vd"))
    elif step == "reconstruct-dual-ac":
        rid = run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
        print(f"=== Dual VD recon Set A+C imagery (grids only) subj={subj} run={rid} ===")
        print(reconstruct.remote(subj, "imagery_set_ac", rid, 3, 0.6, 20, "vd"))
    elif step == "reconstruct-kandinsky":
        rid = run_id or "20260602_034428_train_contrastive_retrieval_subj01_150ep"
        print(f"=== Kandinsky 2.2 recon + CLIP 2WC subj={subj} run={rid} ===")
        print(reconstruct.remote(
            subj, "both", rid, 3, 0.6, 20,
            "kandinsky", "4H_CTR", "best_retrieval", 3, 0.25, 0.3,
        ))
    elif step == "vdvae-encode":
        print(f"=== VDVAE encode {'all subjects' if all_subjects else subj} ===")
        print(vdvae_encode.remote(subj, all_subjects, False))
    elif step == "vdvae-regression":
        vdvae_alpha = 50000.0 if alpha == 1e4 else alpha
        print("=== VDVAE ridge on unaveraged trials (betas_flat) ===")
        print(vdvae_ref_dims.remote())
        if all_subjects:
            print(vdvae_regression_all.remote(run_id, vdvae_alpha))
        else:
            print(vdvae_regression_subj.remote(subj, run_id, vdvae_alpha))
    elif step == "vdvae-ref-dims":
        print(vdvae_ref_dims.remote())
    elif step == "vdvae-dual":
        rid = run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
        print(f"=== VDVAE dual recon subj={subj} run={rid} ===")
        print(reconstruct_vdvae_dual.remote(
            subj, rid, "4H_CTR2", "best_retrieval", 3, 0.75, 50, 7.5,
        ))
    elif step == "vdvae-dual-perception":
        rid = run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
        print(f"=== VDVAE dual recon (perception val only) subj={subj} run={rid} ===")
        print(reconstruct_vdvae_dual.remote(
            subj, rid, "4H_CTR2", "best_retrieval", 3, 0.75, 50, 7.5, True,
        ))
    elif step == "vdvae-dual-imagery-adapted":
        rid = run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        print(f"=== VDVAE imagery Set-B + beta adapter subj={subj} adapter={beta_rid} ===")
        print(reconstruct_vdvae_dual.remote(
            subj, rid, "4H_CTR2", "best_retrieval", 3, 0.75, 50, 7.5,
            False, False, beta_rid, "projector",
        ))
    elif step == "train-diffusion-prior":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        print(f"=== Train diffusion prior subj={subj} ===")
        print(train_diffusion_prior_subj.remote(subj, run_id, dual_rid, epochs, 128))
    elif step == "prior-adapted-recon":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        text_src = text_source or os.environ.get("TEXT_SOURCE", "")
        vd_mix = float(os.environ.get("VD_VISION_MIX", "0.6"))
        osd = os.environ.get("PRIOR_RECON_OUT", "prior_adapted_rerun")
        if text_src == "mlp_text_projector" and "_textmlp" not in osd:
            osd = f"{osd}_textmlp" if osd else "prior_adapted_textmlp"
        if text_src == "mlp_text_hf_dual" and "_texthf" not in osd:
            osd = f"{osd}_texthf" if osd else "prior_adapted_texthf"
        if text_src == "caption_retrieval" and "_captret" not in osd:
            osd = f"{osd}_captret" if osd else "prior_adapted_captret"
        pseed = int(os.environ.get("PRIOR_SEED", "9"))
        vseed = int(os.environ.get("VD_SEED", "9"))
        txt_str = float(os.environ.get("TEXT_TO_IMAGE_STRENGTH", "-1"))
        print(
            f"=== Recon: imagery Set-B (adapter + VDVAE + prior→VD"
            f"{'+ BD text_mlp' if text_src == 'mlp_text_projector' else '+ HF text_mlp' if text_src == 'mlp_text_hf_dual' else '+ caption_retrieval' if text_src == 'caption_retrieval' else ''}) "
            f"subj={subj} → {osd} seeds=({pseed},{vseed}) ==="
        )
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_strength, 50, 7.5,
            False, True, beta_rid, "prior", prior_rid, prior_temperature, True, osd,
            text_src, vd_mix, txt_str, 0.85, pseed, vseed,
        ))
    elif step == "prior-kneeland-hf":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        osd = os.environ.get("PRIOR_RECON_OUT", "prior_kneeland_hf_s9")
        pseed = int(os.environ.get("PRIOR_SEED", "9"))
        vseed = int(os.environ.get("VD_SEED", "9"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        print(
            f"=== Kneeland HF (A=projector/VD0.12, B=prior/VD0.3; no Set C) "
            f"τ={tau} seed={pseed} subj={subj} → {osd} ==="
        )
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, 0.3, 50, 7.5,
            False, True, beta_rid, "prior", prior_rid, tau, True, osd,
            "", 0.6, -1.0, 0.85, pseed, vseed, "kneeland",
        ))
    elif step == "prior-setb-push60":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        seed = int(os.environ.get("PRIOR_SEED", "9"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        jobs = [
            ("prior_vd02", dual_rid, "4H_CTR2", "prior", 0.2, tau),
            ("prior_vd025", dual_rid, "4H_CTR2", "prior", 0.25, tau),
            ("prior_vd03", dual_rid, "4H_CTR2", "prior", 0.3, tau),
            ("proj_vd03", dual_rid, "4H_CTR2", "projector", 0.3, 1.0),
        ]
        print(f"=== Set-B push→60% sweep (4H_CTR2 MLP only, seed={seed}) subj={subj} ===")
        handles = []
        for osd, rid, variant, s2src, vd_s, t in jobs:
            handles.append((
                osd,
                reconstruct_vdvae_dual.spawn(
                    subj, rid, variant, "best_retrieval", 3, vd_s, 50, 7.5,
                    False, True, beta_rid, s2src, prior_rid, t, True, f"setb_{osd}_s{seed}",
                    "", 0.6, -1.0, 0.85, seed, seed, "set_b",
                ),
            ))
        for label, h in handles:
            print(f"  {label}: {h.get()}")
    elif step == "push60-seeds":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        vd_s = float(os.environ.get("VD_STRENGTH", "0.3"))
        seeds = [int(x) for x in os.environ.get("PRIOR_SEEDS", "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31").split(",")]
        s2src = os.environ.get("STAGE2_SOURCE", "prior")
        blend_w = float(os.environ.get("PRIOR_BLEND_WEIGHT", "0.85"))
        print(
            f"=== Push 60%: dual 4H_CTR2 + HF, Set-B only, {s2src} τ={tau} VD={vd_s} "
            f"seeds={len(seeds)} subj={subj} ==="
        )
        handles = []
        for s in seeds:
            osd = f"push60_{s2src}_s{s}"
            handles.append((
                s,
                reconstruct_vdvae_dual.spawn(
                    subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_s, 50, 7.5,
                    False, True, beta_rid, s2src, prior_rid, tau, True, osd,
                    "", 0.6, -1.0, 0.85, s, s, "set_b", blend_w,
                ),
            ))
        best_s, best_k2 = -1, -1.0
        for s, h in handles:
            ret = h.get()
            k2 = -1.0
            if isinstance(ret, str) and "KNEELAND_2WC_RESULT=" in ret:
                k2 = float(ret.split("=", 1)[1])
            print(f"  seed={s}: {ret}")
            if k2 > best_k2:
                best_k2, best_s = k2, s
        print(f"\n=== BEST seed={best_s}  Kneeland 2WC={best_k2:.4f} ({100*best_k2:.1f}%) ===")
    elif step == "push60-refine":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        seed = int(os.environ.get("PRIOR_SEED", "9"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        blends = [float(x) for x in os.environ.get("BLEND_WEIGHTS", "0.7,0.85,0.95,1.0").split(",")]
        vds = [float(x) for x in os.environ.get("VD_STRENGTHS", "0.25,0.3,0.35").split(",")]
        print(f"=== Push 60% refine: prior_blend × VD grid, seed={seed} subj={subj} ===")
        handles = []
        for w in blends:
            for vd_s in vds:
                tag = f"blend{str(w).replace('.','')}_vd{str(vd_s).replace('.','')}"
                handles.append((
                    tag,
                    reconstruct_vdvae_dual.spawn(
                        subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_s, 50, 7.5,
                        False, True, beta_rid, "prior_blend", prior_rid, tau, True,
                        f"push60_{tag}_s{seed}", "", 0.6, -1.0, 0.85, seed, seed, "set_b", w,
                    ),
                ))
        best_tag, best_k2 = "", -1.0
        for tag, h in handles:
            ret = h.get()
            k2 = -1.0
            if isinstance(ret, str) and "KNEELAND_2WC_RESULT=" in ret:
                k2 = float(ret.split("=", 1)[1])
            print(f"  {tag}: {ret}")
            if k2 > best_k2:
                best_k2, best_tag = k2, tag
        print(f"\n=== BEST {best_tag}  Kneeland 2WC={best_k2:.4f} ({100*best_k2:.1f}%) ===")
    elif step == "quality-s29":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        jobs = [
            ("prior_vd015", "prior", 0.15, tau, None),
            ("prior_vd020", "prior", 0.20, tau, None),
            ("prior_vd025", "prior", 0.25, tau, None),
            ("prior_vd030", "prior", 0.30, tau, None),
            ("blend50_vd020", "prior_blend", 0.20, tau, 0.5),
            ("blend50_vd025", "prior_blend", 0.25, tau, 0.5),
            ("proj_vd012", "projector", 0.12, 1.0, None),
        ]
        print(
            f"=== Quality sweep @ seed={seed}: 4H_CTR2/best_retrieval MLP only "
            f"(prior | prior_blend | projector), no joint_ridge ==="
        )
        handles = []
        for osd, s2src, vd_s, t, blend_w in jobs:
            spawn_kw = dict(
                subj=subj,
                run_id=dual_rid,
                variant="4H_CTR2",
                ckpt_prefer="best_retrieval",
                n_perception=3,
                vd_strength=vd_s,
                vd_steps=50,
                guidance_scale=7.5,
                perception_only=False,
                imagery_only=True,
                beta_adapter_run_id=beta_rid,
                stage2_source=s2src,
                prior_run_id=prior_rid,
                prior_temperature=t,
                eval_recon_kneeland=True,
                out_subdir=f"quality_{osd}_s{seed}",
                text_source="",
                vd_vision_mix=0.6,
                text_to_image_strength=-1.0,
                brain_token_weight=0.85,
                prior_seed=seed,
                vd_seed=seed,
                imagery_sets="set_b",
                prior_blend_weight=blend_w if blend_w is not None else 0.85,
            )
            handles.append((osd, reconstruct_vdvae_dual.spawn(**spawn_kw)))
        best: tuple[str, float, float] = ("", -1.0, -1.0)
        for tag, h in handles:
            ret = str(h.get())
            k2, cg = -1.0, -1.0
            for line in ret.splitlines():
                if "KNEELAND_2WC_RESULT=" in line:
                    k2 = float(line.rsplit("=", 1)[-1])
                if "CLIP_GT_COSIM=" in line:
                    cg = float(line.rsplit("=", 1)[-1])
            combo = (0.4 * k2 + 0.6 * cg) if k2 >= 0 and cg >= 0 else -1.0
            print(f"  {tag}: k2={k2:.3f} clip_gt={cg:.3f} combo={combo:.3f}")
            if combo > 0.4 * best[1] + 0.6 * best[2]:
                best = (tag, k2, cg)
        print(f"\n=== BEST quality config: {best[0]}  Kneeland={best[1]:.1%}  CLIP→GT={best[2]:.3f} ===")
    elif step == "kneeland-ab-sweep":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        jobs = [
            ("kn_vd03", 0.3, "", -1.0),
            ("kn_vd025", 0.25, "", -1.0),
            ("kn_vd02", 0.2, "", -1.0),
            ("kn_vd03_t005", 0.3, "mlp_text_hf_dual", 0.05),
            ("kn_vd03_t008", 0.3, "mlp_text_hf_dual", 0.08),
            ("kn_vd025_t005", 0.25, "mlp_text_hf_dual", 0.05),
        ]
        print(
            f"=== Kneeland A+B sweep (4H_CTR2 MLP, A=proj/0.12, B=prior τ={tau}, seed={seed}) ==="
        )
        handles = []
        for tag, vd_b, txt, txt_str in jobs:
            osd = f"kn_ab_{tag}_s{seed}"
            handles.append((
                tag,
                reconstruct_vdvae_dual.spawn(
                    subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_b, 50, 7.5,
                    False, False, beta_rid, "prior", prior_rid, tau, True, osd,
                    txt, 0.6, txt_str, 0.85, seed, seed, "kneeland", 0.85, vd_b,
                ),
            ))
        best: tuple[str, float, float, float, float] = ("", -1.0, -1.0, -1.0, -1.0)
        for tag, h in handles:
            ret = str(h.get())
            k_avg, k_a, k_b, cg = -1.0, -1.0, -1.0, -1.0
            for line in ret.splitlines():
                if line.startswith("KNEELAND_2WC_AVG_AB="):
                    k_avg = float(line.rsplit("=", 1)[-1])
                if line.startswith("KNEELAND_2WC_SET_A="):
                    k_a = float(line.rsplit("=", 1)[-1])
                if line.startswith("KNEELAND_2WC_SET_B="):
                    k_b = float(line.rsplit("=", 1)[-1])
                if line.startswith("CLIP_GT_COSIM="):
                    cg = float(line.rsplit("=", 1)[-1])
            combo = (0.45 * k_avg + 0.55 * cg) if k_avg >= 0 and cg >= 0 else -1.0
            print(
                f"  {tag}: avg_AB={k_avg:.3f} A={k_a:.3f} B={k_b:.3f} "
                f"clip_gt={cg:.3f} combo={combo:.3f}"
            )
            if combo > 0.45 * best[1] + 0.55 * best[2]:
                best = (tag, k_avg, cg, k_a, k_b)
        print(
            f"\n=== BEST {best[0]}  A+B avg={best[1]:.1%}  CLIP→GT={best[2]:.3f}  "
            f"(A={best[3]:.1%} B={best[4]:.1%}) ==="
        )
    elif step == "kneeland-a-text-b-prior":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        vd_b = float(os.environ.get("VD_STRENGTH", "0.3"))
        txt_strengths = [
            float(x) for x in os.environ.get("HF_TEXT_STRENGTHS", "0.05,0.08,0.1").split(",")
        ]
        print(
            f"=== Kneeland A=text (MLP) B=prior/vision-only "
            f"(τ={tau} VD_B={vd_b} seed={seed}) ==="
        )
        handles = []
        for t in txt_strengths:
            tag = f"a_txt{str(t).replace('.', '')}_b_vd{str(vd_b).replace('.', '')}"
            osd = f"kn_atxt_{tag}_s{seed}"
            handles.append((
                tag,
                reconstruct_vdvae_dual.spawn(
                    subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_b, 50, 7.5,
                    False, False, beta_rid, "prior", prior_rid, tau, True, osd,
                    "", 0.6, t, 0.85, seed, seed, "kneeland", 0.85, vd_b,
                    "mlp_text_hf_dual", "",
                ),
            ))
        best: tuple[str, float, float, float, float] = ("", -1.0, -1.0, -1.0, -1.0)
        for tag, h in handles:
            ret = str(h.get())
            k_avg, k_a, k_b, cg = -1.0, -1.0, -1.0, -1.0
            for line in ret.splitlines():
                if line.startswith("KNEELAND_2WC_AVG_AB="):
                    k_avg = float(line.rsplit("=", 1)[-1])
                if line.startswith("KNEELAND_2WC_SET_A="):
                    k_a = float(line.rsplit("=", 1)[-1])
                if line.startswith("KNEELAND_2WC_SET_B="):
                    k_b = float(line.rsplit("=", 1)[-1])
                if line.startswith("CLIP_GT_COSIM="):
                    cg = float(line.rsplit("=", 1)[-1])
            combo = (0.45 * k_avg + 0.55 * cg) if k_avg >= 0 and cg >= 0 else -1.0
            print(
                f"  {tag}: avg_AB={k_avg:.3f} A={k_a:.3f} B={k_b:.3f} "
                f"clip_gt={cg:.3f} combo={combo:.3f}"
            )
            if combo > 0.45 * best[1] + 0.55 * best[2]:
                best = (tag, k_avg, cg, k_a, k_b)
        print(
            f"\n=== BEST {best[0]}  A+B avg={best[1]:.1%}  CLIP→GT={best[2]:.3f}  "
            f"(A={best[3]:.1%} B={best[4]:.1%}) ==="
        )
    elif step == "prior-set-a-recon":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        osd = os.environ.get("PRIOR_RECON_OUT", "prior_set_a_vd012_s9")
        pseed = int(os.environ.get("PRIOR_SEED", "9"))
        vseed = int(os.environ.get("VD_SEED", "9"))
        print(f"=== Set A only (projector, VD=0.12) seed={pseed} subj={subj} → {osd} ===")
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, 0.12, 50, 7.5,
            False, True, beta_rid, "projector", prior_rid, 1.0, True, osd,
            "", 0.6, -1.0, 0.85, pseed, vseed, "set_a",
        ))
    elif step == "prior-text-sweep":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        seed = int(os.environ.get("PRIOR_SEED", "9"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        vd_s = float(os.environ.get("VD_STRENGTH", "0.3"))
        hf_txt = [float(x) for x in os.environ.get("HF_TEXT_STRENGTHS", "0.1,0.15,0.2").split(",")]
        print(
            f"=== HF text ablation (τ={tau}, VD={vd_s}, seed={seed}) txt={hf_txt} subj={subj} ==="
        )
        handles = []
        for t in hf_txt:
            osd = f"prior_vd03_t15_texthf_t{str(t).replace('.', '')}_s{seed}"
            handles.append((
                f"hf_txt={t}",
                reconstruct_vdvae_dual.spawn(
                    subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_s, 50, 7.5,
                    False, True, beta_rid, "prior", prior_rid, tau, True, osd,
                    "mlp_text_hf_dual", 0.6, t, 0.85, seed, seed,
                ),
            ))
        for label, h in handles:
            print(f"  {label}: {h.get()}")
    elif step == "prior-adapted-vd-ablation":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        osd = f"prior_adapted_vd{str(vd_strength).replace('.', '')}"
        print(
            f"=== VD ablation: strength={vd_strength} (low=CLIP-heavy) "
            f"imagery Set-B subj={subj} → {osd} ==="
        )
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_strength, 50, 7.5,
            False, True, beta_rid, "prior", prior_rid, prior_temperature, False, osd,
        ))
    elif step == "prior-adapted-vd-sweep":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        strengths = [
            float(x) for x in os.environ.get("VD_STRENGTHS", "0.2,0.3,0.4").split(",")
        ]
        print(f"=== VD strength sweep {strengths} (prior→VD, imagery Set-B) subj={subj} ===")
        handles = []
        for s in strengths:
            osd = f"prior_adapted_vd{str(s).replace('.', '')}"
            handles.append((
                s,
                reconstruct_vdvae_dual.spawn(
                    subj, dual_rid, "4H_CTR2", "best_retrieval", 3, s, 50, 7.5,
                    False, True, beta_rid, "prior", prior_rid, prior_temperature, False, osd,
                ),
            ))
        for s, h in handles:
            print(f"  strength={s}: {h.get()}")
    elif step == "prior-temp-sweep":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        vd_s = float(os.environ.get("VD_STRENGTH", str(vd_strength)))
        temps = [
            float(x) for x in os.environ.get("PRIOR_TEMPS", "1.0,1.5,2.0,3.0").split(",")
        ]
        print(
            f"=== Prior DDIM temperature sweep {temps} "
            f"(VD={vd_s}, imagery Set-B) subj={subj} ==="
        )
        handles = []
        for t in temps:
            tag = str(t).replace(".", "")
            osd = f"prior_adapted_vd{str(vd_s).replace('.', '')}_t{tag}"
            handles.append((
                t,
                reconstruct_vdvae_dual.spawn(
                    subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_s, 50, 7.5,
                    False, True, beta_rid, "prior", prior_rid, t, True, osd,
                ),
            ))
        for t, h in handles:
            print(f"  τ={t}: {h.get()}")
    elif step == "stage1-diagnose":
        clip_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        vdvae_rid = os.environ.get(
            "VDVAE_ADAPTER_RUN", "20260604_060546_imagery_vdvae_beta_adapter_subj01",
        )
        full = os.environ.get("STAGE1_FULL_LAYERS", "0") == "1"
        osd = os.environ.get("STAGE1_OUT", "stage1_diagnose")
        print(f"=== Stage-1 diagnose (Set-B W/K/B/C/D) → {osd} ===")
        print(stage1_diagnose.remote(subj, clip_rid, vdvae_rid, full, osd))
    elif step == "semantics-compare":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        pseed = int(os.environ.get("PRIOR_SEED", "29"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        osd = os.environ.get("PRIOR_RECON_OUT", f"semantics_compare_s{pseed}")
        print(f"=== Set-B semantics grid (unCLIP + VD) seed={pseed} → {osd} ===")
        print(semantics_compare.remote(
            subj, dual_rid, beta_rid, prior_rid, tau, pseed, osd,
        ))
    elif step == "kneeland-multisample-atxt":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        seeds = os.environ.get("MULTISAMPLE_SEEDS", "20,21,22,23,24,25,26,27,28,29")
        txt = float(os.environ.get("HF_TEXT_STRENGTH", "0.05"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        vd_b = float(os.environ.get("VD_STRENGTH", "0.3"))
        out_base = os.environ.get("PRIOR_RECON_OUT", "kn_atxt_ms")
        print(f"=== 10-sample Kneeland multisample (A=text, B=prior) seeds={seeds} ===")
        print(kneeland_multisample_atxt.remote(
            subj, seeds, dual_rid, beta_rid, prior_rid, txt, tau, vd_b, out_base, 1000,
        ))
    elif step == "eval-kneeland-paper":
        dirs = os.environ.get(
            "RUN_DIRS",
            "kn_atxt_a_txt005_b_vd03_s29,kn_ab_kn_vd03_s29,push60_prior_s29",
        )
        seed = int(os.environ.get("PRIOR_SEED", "42"))
        print(f"=== Paper vs legacy Kneeland 2WC on: {dirs} ===")
        print(eval_kneeland_paper_runs.remote(subj, dirs, 1000, seed))
    elif step == "reeval-prior-grid":
        grid_rel = os.environ.get(
            "GRID_REL",
            "reconstructions/subj01/prior_adapted_vd03_t15/imagery_setB_grid.png",
        )
        stage2_rel = os.environ.get("STAGE2_DIR", "")
        label = os.environ.get("GRID_LABEL", "prior_adapted_vd03_t15")
        exp = float(os.environ.get("EXPECTED_K2WC", "0.518"))
        src = stage2_rel or grid_rel
        print(f"=== Re-eval Kneeland on {src} subj={subj} ===")
        print(reeval_recon_grid.remote(subj, grid_rel, stage2_rel, 1000, label, exp))
    elif step == "prior-adapted-t15-restore":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        osd = os.environ.get("PRIOR_RECON_OUT", "prior_adapted_vd03_t15_restore")
        pseed = int(os.environ.get("PRIOR_SEED", "9"))
        vseed = int(os.environ.get("VD_SEED", "9"))
        print(
            f"=== Restore 51.8% pipeline: HF image-variation legacy, VD=0.3, τ=1.5 → {osd} ==="
        )
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, 0.3, 50, 7.5,
            False, True, beta_rid, "prior", prior_rid, 1.5, True, osd,
            "", 0.6, -1.0, 0.85, pseed, vseed,
        ))
        stage2_rel = f"reconstructions/{subj}/{osd}/stage2_setB"
        print(f"=== Re-eval Kneeland on {stage2_rel} (expected 0.518) ===")
        print(reeval_recon_grid.remote(subj, "", stage2_rel, 1000, osd, 0.518))
    elif step == "prior-seed-sweep-recon":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        seeds = [int(x) for x in os.environ.get("PRIOR_SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]
        print(f"=== Prior+VD seed sweep {seeds} (HF legacy, VD=0.3, τ=1.5) subj={subj} ===")
        handles = []
        for s in seeds:
            osd = f"prior_adapted_vd03_t15_s{s}"
            handles.append((
                s,
                reconstruct_vdvae_dual.spawn(
                    subj, dual_rid, "4H_CTR2", "best_retrieval", 3, 0.3, 50, 7.5,
                    False, True, beta_rid, "prior", prior_rid, 1.5, True, osd,
                    "", 0.6, -1.0, 0.85, s, s,
                ),
            ))
        for s, h in handles:
            print(f"  seed={s}: {h.get()}")
    elif step == "prior-imageonly-verify":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        osd = os.environ.get("PRIOR_RECON_OUT", "prior_adapted_verify_t15")
        print(
            f"=== HF image-only prior verify (VD={vd_strength}, τ={prior_temperature}) "
            f"subj={subj} → {osd} ==="
        )
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_strength, 50, 7.5,
            False, True, beta_rid, "prior", prior_rid, prior_temperature, True, osd,
        ))
    elif step == "prior-adapted-pipeline":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        prior_rid = run_id or ""
        if os.environ.get("SKIP_PRIOR_TRAIN", "1") == "1":
            print("=== Prior training skipped (use SKIP_PRIOR_TRAIN=0 to retrain) ===")
        else:
            print("=== Train diffusion prior (perception) ===")
            print(train_diffusion_prior_subj.remote(subj, prior_rid, dual_rid, epochs, 128))
        print("=== Recon: imagery Set-B only + recon Kneeland ===")
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, 0.75, 50, 7.5,
            False, True, beta_rid, "prior", prior_rid, prior_temperature, True, "",
        ))
    elif step == "vdvae-pipeline":
        rid = run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
        print("=== VDVAE full pipeline: encode → trial ridge → dual recon ===")
        print(vdvae_encode.remote(subj, all_subjects, False))
        if all_subjects:
            print(vdvae_regression_all.remote(run_id, 50000.0))
        else:
            print(vdvae_regression_subj.remote(subj, run_id, 50000.0))
        print(reconstruct_vdvae_dual.remote(
            subj, rid, "4H_CTR2", "best_retrieval", 3, 0.75, 50, 7.5,
        ))
    elif step == "reconstruct-setb-clip2wc":
        rid = run_id or "20260602_034428_train_contrastive_retrieval_subj01_150ep"
        prefer = ckpt_prefer if ckpt_prefer != "final" else "best_clip"
        print(f"=== Set-B imagery recon + CLIP 2WC subj={subj} run={rid} ===")
        print(reconstruct_imagery_small.remote(
            subj, 3, 0.6, rid, prefer, "4H_CTR", "regression", True, True,
        ))
    elif step == "reconstruct-imagery":
        print(reconstruct_imagery_small.remote(
            subj, 3, 0.6, run_id, ckpt_prefer, variant, clip_source,
        ))
    elif step == "reconstruct-perception":
        rid = run_id or (
            "20260602_034428_train_contrastive_retrieval_subj01_150ep"
            if variant == "4H_CTR"
            else "20260601_052537_varA_subj01_150ep"
        )
        print(reconstruct_perception_small.remote(
            subj, 3, 0.6, rid, ckpt_prefer, variant, clip_source,
        ))
    elif step == "reconstruct-setb-4h-ctr":
        rid = run_id or "20260602_034428_train_contrastive_retrieval_subj01_150ep"
        prefer = ckpt_prefer if ckpt_prefer != "final" else "best_retrieval"
        print(f"=== Set-B recon 4H_CTR run={rid} ckpt={prefer} clip={clip_source} ===")
        h_imagery = reconstruct_imagery_small.spawn(
            subj, 3, 0.6, rid, prefer, "4H_CTR", clip_source,
        )
        h_perception = reconstruct_perception_small.spawn(
            subj, 3, 0.6, rid, prefer, "4H_CTR", clip_source,
        )
        print("Imagery:", h_imagery.get())
        print("Perception:", h_perception.get())
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
    elif step == "train-imagery-adapter":
        rid = run_id or ""
        asp = os.environ.get("ADAPTER_ALIGN_SPACE", "mlp")
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        bw = float(os.environ.get("BETA_WEIGHT", "0.35"))
        init_rid = beta_adapter_run_id or os.environ.get("INIT_ADAPTER_RUN", "")
        print(f"=== Imagery beta adapter subj={subj} align={asp} ===")
        h = train_imagery_beta_adapter_subj.spawn(
            subj, rid, epochs, asp, dual_rid, bw, init_rid, "",
        )
        if "--detach" in sys.argv or "-d" in sys.argv:
            print(
                "Spawned adapter training on Modal (detached). "
                "Do not stop the app from the dashboard until LOO + final fit complete."
            )
            print(f"  function_call_id={h.object_id}")
        else:
            print(h.get())
    elif step == "eval-beta-alignment":
        rid = beta_adapter_run_id or os.environ.get(
            "BETA_ADAPTER_RUN", "20260603_230807_imagery_beta_adapter_subj01",
        )
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        print(f"=== Beta alignment report adapter={rid} ===")
        print(eval_beta_alignment_subj.remote(subj, rid, dual_rid))
    elif step in ("kneeland-atxt-recon", "kneeland-hybrid-recon", "kneeland-honest-recon", "kneeland-calibrated-recon"):
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        if step == "kneeland-calibrated-recon":
            no_adapter = (
                os.environ.get("NO_ADAPTER", "0") == "1"
                or beta_adapter_run_id in ("none", "off", "false", "0")
            )
            if no_adapter:
                beta_rid = ""
                osd_default = "kn_atxt_a_txt005_b_vd03_s29_no_adapter"
            else:
                beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
                osd_default = "kn_atxt_a_txt005_b_vd03_s29"
        elif step == "kneeland-honest-recon":
            beta_rid = beta_adapter_run_id or os.environ.get("HONEST_ADAPTER_RUN", "")
            if not beta_rid:
                raise SystemExit(
                    "Run train-imagery-adapter-honest first, then pass "
                    "--beta-adapter-run-id or HONEST_ADAPTER_RUN"
                )
            osd_default = "kn_atxt_honest_s29"
        else:
            beta_rid = beta_adapter_run_id or os.environ.get("BETA_ADAPTER_RUN", "")
            if not beta_rid:
                raise SystemExit("Set --beta-adapter-run-id or BETA_ADAPTER_RUN")
            osd_default = f"kn_atxt_s{int(os.environ.get('PRIOR_SEED', '29'))}"
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        vd_b = float(os.environ.get("VD_STRENGTH", "0.3"))
        txt = float(os.environ.get("HF_TEXT_STRENGTH", "0.05"))
        s1_src = os.environ.get("STAGE1_SOURCE", "vdvae_ridge")
        osd = os.environ.get("KNEELAND_OUT", osd_default)
        label = "calibrated" if step == "kneeland-calibrated-recon" else (
            "honest" if step == "kneeland-honest-recon" else "custom"
        )
        adapter_label = "(none)" if not beta_rid else beta_rid
        print(
            f"=== Kneeland A+B recon ({label}) adapter={adapter_label} "
            f"(stage1={s1_src} txt={txt} VD_B={vd_b} seed={seed}) → {osd} ==="
        )
        # Match kneeland-a-text-b-prior (65% run): global text off; HF text on Set A only.
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_b, 50, 7.5,
            False, False, beta_rid, "prior", "", tau, True, osd,
            "", 0.6, txt, 0.85, seed, seed, "kneeland", 0.85, vd_b,
            "mlp_text_hf_dual", "", s1_src,
        ))
    elif step == "kneeland-cross-subject-recon":
        import re

        train_s = os.environ.get("ADAPTER_TRAIN_SUBJ", "subj01")
        dual_base = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        dual_rid = (
            re.sub(r"subj\d+", subj, dual_base, count=1)
            if re.search(r"subj\d+", dual_base)
            else dual_base
        )
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        vd_b = float(os.environ.get("VD_STRENGTH", "0.3"))
        txt = float(os.environ.get("HF_TEXT_STRENGTH", "0.05"))
        s1_src = os.environ.get("STAGE1_SOURCE", "vdvae_ridge")
        pooled = os.environ.get("USE_POOLED_PRIOR", "0") == "1"
        osd = os.environ.get(
            "KNEELAND_OUT",
            f"kn_xsub_{train_s}_to_{subj}_s{seed}" + ("_pooled" if pooled else ""),
        )
        print(
            f"=== Kneeland cross-subject: adapter={train_s} infer={subj} "
            f"dual={dual_rid} prior={'pooled' if pooled else subj} → {osd} ==="
        )
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_b, 50, 7.5,
            False, False, beta_rid, "prior", "", tau, True, osd,
            "", 0.6, txt, 0.85, seed, seed, "kneeland", 0.85, vd_b,
            "mlp_text_hf_dual", "", s1_src, train_s, pooled,
        ))
    elif step == "kneeland-no-adapter-fair":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        vd_b = float(os.environ.get("VD_STRENGTH", "0.3"))
        txt = float(os.environ.get("HF_TEXT_STRENGTH", "0.05"))
        s1_src = os.environ.get("STAGE1_SOURCE", "vdvae_ridge")
        osd = os.environ.get(
            "KNEELAND_OUT", f"kn_no_adapter_{s1_src}_txt{str(txt).replace('.', '')}_s{seed}",
        )
        print(
            f"=== Kneeland A+B no adapter (fair): stage1={s1_src} "
            f"A=text@{txt} B=prior/vision τ={tau} VD_B={vd_b} seed={seed} → {osd} ==="
        )
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_b, 50, 7.5,
            False, False, "", "prior", "", tau, True, osd,
            "", 0.6, txt, 0.85, seed, seed, "kneeland", 0.85, vd_b,
            "mlp_text_hf_dual", "", s1_src,
        ))
    elif step == "kneeland-no-adapter-text-sweep":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        vd_b = float(os.environ.get("VD_STRENGTH", "0.3"))
        s1_src = os.environ.get("STAGE1_SOURCE", "vdvae_ridge")
        txt_strengths = [
            float(x) for x in os.environ.get("HF_TEXT_STRENGTHS", "0.05,0.08,0.1,0.12,0.15").split(",")
        ]
        print(
            f"=== No-adapter text sweep (Set A HF text): stage1={s1_src} "
            f"strengths={txt_strengths} seed={seed} ==="
        )
        best: tuple[str, float] = ("", -1.0)
        for t in txt_strengths:
            tag = str(t).replace(".", "")
            osd = f"kn_no_adapter_{s1_src}_txt{tag}_s{seed}"
            ret = str(reconstruct_vdvae_dual.remote(
                subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_b, 50, 7.5,
                False, False, "", "prior", "", tau, True, osd,
                "", 0.6, t, 0.85, seed, seed, "kneeland", 0.85, vd_b,
                "mlp_text_hf_dual", "", s1_src,
            ))
            k_avg = -1.0
            for line in ret.splitlines():
                if line.startswith("KNEELAND_2WC_PAPER_AVG_AB="):
                    k_avg = float(line.rsplit("=", 1)[-1])
            print(ret)
            if k_avg > best[1]:
                best = (osd, k_avg)
        print(f"\n=== BEST no-adapter text run: {best[0]}  paper A+B 2WC={best[1]:.4f} ({100*best[1]:.1f}%) ===")
    elif step == "train-imagery-adapter-honest":
        rid = run_id or ""
        asp = os.environ.get("ADAPTER_ALIGN_SPACE", "mlp")
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        bw = float(os.environ.get("BETA_WEIGHT", "0.35"))
        print("=== Honest adapter: final fit on Set C only (concepts; A/B held out) ===")
        h = train_imagery_beta_adapter_subj.spawn(
            subj, rid, epochs, asp, dual_rid, bw, "", "A,B", False,
        )
        if "--detach" in sys.argv or "-d" in sys.argv:
            print("Spawned detached. Do not stop the app in the dashboard until LOO + fit complete.")
            print(f"  function_call_id={h.object_id}")
        else:
            ret = h.get()
            print(ret)
            print(
                "\n  Next: modal run modal_app.py --step kneeland-honest-recon --subj subj01 "
                "--beta-adapter-run-id <run_id from above>"
            )
    elif step == "train-imagery-adapter-hybrid":
        rid = run_id or ""
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        bw = float(os.environ.get("BETA_WEIGHT", "0.35"))
        init_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        print(f"=== Hybrid adapter (MLP+{bw:.0%} beta) warm-start={init_rid} ===")
        h = train_imagery_beta_adapter_subj.spawn(
            subj, rid, epochs, "hybrid", dual_rid, bw, init_rid, "",
        )
        if "--detach" in sys.argv or "-d" in sys.argv:
            print(
                "Spawned adapter training on Modal (detached). "
                "Do not stop the app from the dashboard until LOO + final fit complete."
            )
            print(f"  function_call_id={h.object_id}")
        else:
            print(h.get())
    elif step == "kneeland-adapter-ablation":
        print("=== Calibrated baseline (existing) ===")
        print("  kn_atxt_a_txt005_b_vd03_s29  (~65% A+B paper)")
        print("=== Honest adapter LOO + Set-C fit + Kneeland recon ===")
        rid = run_id or ""
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        osd = os.environ.get("KNEELAND_OUT", f"kn_atxt_honest_s{seed}")
        vd_b = float(os.environ.get("VD_STRENGTH", "0.3"))
        txt = float(os.environ.get("HF_TEXT_STRENGTH", "0.05"))
        skip_loo = os.environ.get("SKIP_LOO", "0") == "1"
        args_pipe = (subj, rid, epochs, skip_loo, seed, vd_b, txt, osd)
        if "--detach" in sys.argv or "-d" in sys.argv:
            h = honest_kneeland_pipeline_subj.spawn(*args_pipe)
            print("Spawned detached on Modal (LOO + train + recon). Do not stop the app in the dashboard.")
            print(f"  function_call_id={h.object_id}")
            print("  Monitor: modal app logs  |  Outputs when done:")
            print(f"    runs/<timestamp>_imagery_beta_adapter_mlp_honest_{subj}/")
            print(f"    reconstructions/{subj}/{osd}/")
            print(f"    results/vdvae_recon_{subj}_{osd}.json")
        else:
            print(honest_kneeland_pipeline_subj.remote(*args_pipe))
    elif step == "kneeland-honest-pipeline":
        rid = run_id or ""
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        osd = os.environ.get("KNEELAND_OUT", f"kn_atxt_honest_s{seed}")
        vd_b = float(os.environ.get("VD_STRENGTH", "0.3"))
        txt = float(os.environ.get("HF_TEXT_STRENGTH", "0.05"))
        skip_loo = os.environ.get("SKIP_LOO", "0") == "1"
        args_pipe = (subj, rid, epochs, skip_loo, seed, vd_b, txt, osd)
        if "--detach" in sys.argv or "-d" in sys.argv:
            h = honest_kneeland_pipeline_subj.spawn(*args_pipe)
            print("Spawned detached (honest Kneeland pipeline).")
            print(f"  function_call_id={h.object_id}")
        else:
            print(honest_kneeland_pipeline_subj.remote(*args_pipe))
    elif step == "train-vdvae-beta-adapter":
        clip_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        do_recon = os.environ.get("EVAL_RECON", "0") == "1"
        do_combined = os.environ.get("VDVAE_ADAPTER_COMBINED", "0") == "1"
        print(
            f"=== VDVAE early-layer beta adapter LOO subj={subj} "
            f"init={clip_rid} combined={do_combined} eval_recon={do_recon} ==="
        )
        print(train_imagery_vdvae_beta_adapter_subj.remote(
            subj, run_id, clip_rid, do_combined, do_recon, 20,
        ))
    elif step == "train-vdvae-beta-adapter-full":
        clip_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        print(f"=== VDVAE beta adapter LOO + recon Kneeland eval subj={subj} ===")
        print(train_imagery_vdvae_beta_adapter_subj.remote(
            subj, run_id, clip_rid, False, True, 20,
        ))
    elif step == "train-vdvae-beta-adapter-combined":
        clip_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        do_recon = os.environ.get("EVAL_RECON", "1") == "1"
        print(f"=== VDVAE+CLIP joint beta adapter LOO subj={subj} eval_recon={do_recon} ===")
        print(train_imagery_vdvae_beta_adapter_subj.remote(
            subj, run_id, clip_rid, True, do_recon, 20,
        ))
    elif step == "vdvae-beta-adapter-recon":
        vdvae_rid = beta_adapter_run_id or os.environ.get(
            "VDVAE_ADAPTER_RUN", "20260604_060546_imagery_vdvae_beta_adapter_subj01",
        )
        jr_rid = run_id or "20260601_053733_joint_ridge_all8"
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        vd_jr = float(os.environ.get("VD_STRENGTH_JR", "0.5"))
        vd_pr = float(os.environ.get("VD_STRENGTH", "0.3"))
        osd = os.environ.get("VDVAE_RECON_OUT", "vdvae_beta_adapted")
        print(f"=== VDVAE adapter recon: S1 + JR@{vd_jr} + prior@{vd_pr} adapter={vdvae_rid} ===")
        print(reconstruct_vdvae_dual.remote(
            subj, jr_rid, "joint_ridge", "best_retrieval", 3, vd_jr, 50, 7.5,
            False, True, vdvae_rid, "joint_ridge", "", 1.0, True, f"{osd}_jr05",
        ))
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd_pr, 50, 7.5,
            False, True, vdvae_rid, "prior", "", 1.5, True, f"{osd}_prior03_t15",
        ))
    elif step == "eval-kneeland-three-way":
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        extra = [
            "--joint-ridge-run-id", "20260601_053733_joint_ridge_all8",
            "--dual-run-id", "20260602_053625_train_dual_contrastive_subj01_150ep",
            "--beta-adapter-run-id", beta_rid,
        ]
        if run_id:
            extra.extend(["--mlp-adapter-run-id", run_id])
        print(f"=== Kneeland 2WC (Set A+B) subj={subj} beta_adapter={beta_rid} ===")
        print(train.remote(subj, "eval_kneeland_three_way.py", 1, "", extra))
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
    elif step == "eval-kneeland":
        if variant.lower() in ("prior_adapted", "prior-adapted"):
            dual_rid = run_id or os.environ.get(
                "DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep",
            )
            beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
            extra = [
                "--variant", "prior_adapted",
                "--source", source,
                "--run-id", dual_rid,
                "--subset", "kneeland",
                "--metric", "kneeland",
                "--n-pairs", "1000",
                "--beta-adapter-run-id", beta_rid,
            ]
            print(f"=== Kneeland embedding path (adapter→MLP→prior→CLIP) subj={subj} ===")
            print(train.remote(subj, "eval_imagery_crossdecode.py", 1, dual_rid, extra))
        elif variant.lower() in ("joint_ridge", "jointridge"):
            rid = run_id or "20260601_053733_joint_ridge_all8"
            v = "joint_ridge"
            head = "clip" if eval_head in ("both", "clip", "regression") else eval_head
            extra = [
                "--variant", v,
                "--run-id", rid,
                "--ckpt-prefer", "best_retrieval" if ckpt_prefer == "final" else ckpt_prefer,
                "--subset", "kneeland",
                "--head", head,
                "--metric", "kneeland",
                "--n-pairs", "1000",
            ]
            if beta_adapter_run_id:
                extra.extend(["--beta-adapter-run-id", beta_adapter_run_id])
            print(f"=== Kneeland Table-1 2WC imagery subj={subj} variant={v} head={head} ===")
            print(train.remote(subj, "eval_imagery_crossdecode.py", 1, rid, extra))
        else:
            rid = run_id or "20260602_053625_train_dual_contrastive_subj01_150ep"
            v = variant if variant not in ("A", "4H") else "4H_CTR2"
            head = eval_head
            extra = [
                "--variant", v,
                "--run-id", rid,
                "--ckpt-prefer", "best_retrieval" if ckpt_prefer == "final" else ckpt_prefer,
                "--subset", "kneeland",
                "--head", head,
                "--metric", "kneeland",
                "--n-pairs", "1000",
            ]
            if beta_adapter_run_id:
                extra.extend(["--beta-adapter-run-id", beta_adapter_run_id])
            print(f"=== Kneeland Table-1 2WC imagery subj={subj} variant={v} head={head} ===")
            print(train.remote(subj, "eval_imagery_crossdecode.py", 1, rid, extra))
    elif step == "reconstruct-imagery-unclip":
        rid = run_id or (
            "20260601_053733_joint_ridge_all8"
            if variant.lower() in ("joint_ridge", "jointridge")
            else "20260602_053625_train_dual_contrastive_subj01_150ep"
        )
        sub = imagery_subset if imagery_subset != "set_b" else "set_b"
        print(f"=== Imagery SD unCLIP subj={subj} variant={variant} subset={sub} ===")
        print(reconstruct_imagery_unclip.remote(
            subj, variant, rid, sub, clip_source, ckpt_prefer, 50, 7.5,
            "", False, False,
        ))
    elif step == "joint-ridge-recon-kneeland":
        jr_rid = run_id or "20260601_053733_joint_ridge_all8"
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        vd_s = float(os.environ.get("VD_STRENGTH", "0.5"))
        osd = f"joint_ridge_vdvae_s{str(vd_s).replace('.', '')}"
        print(
            f"=== Brain-Diffuser path: VDVAE S1 + joint-ridge CLIP (62.9%) + VD "
            f"(strength={vd_s}) → recon Kneeland subj={subj} ==="
        )
        print(reconstruct_vdvae_dual.remote(
            subj, jr_rid, "joint_ridge", "best_retrieval", 3, vd_s, 50, 7.5,
            False, True, beta_rid, "joint_ridge", "", 1.0, True, osd,
        ))
    elif step == "eval-imagery-crossdecode":
        rid = run_id or "20260602_034428_train_contrastive_retrieval_subj01_150ep"
        extra = [
            "--variant", variant if variant != "A" else "4H_CTR",
            "--run-id", rid,
            "--ckpt-prefer", ckpt_prefer if ckpt_prefer != "final" else "best_retrieval",
            "--subset", imagery_subset,
            "--head", eval_head,
        ]
        print(f"=== Imagery cross-decode subj={subj} subset={imagery_subset} head={eval_head} ===")
        print(train.remote(subj, "eval_imagery_crossdecode.py", 1, rid, extra))
    elif step == "kneeland-production-pipeline":
        rid = run_id or ""
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        osd = os.environ.get("KNEELAND_OUT", "")
        skip_loo = os.environ.get("SKIP_LOO", "1") == "1"
        args_pipe = (subj, rid, epochs, 200, 150, skip_loo, seed, osd)
        print(
            f"=== Production Kneeland pipeline {subj} "
            f"(dual {epochs}ep, adapter LOO={not skip_loo}) ==="
        )
        if "--detach" in sys.argv or "-d" in sys.argv:
            h = kneeland_production_pipeline_subj.spawn(*args_pipe)
            print("Spawned detached on Modal. Do not stop the app until the job completes.")
            print(f"  function_call_id={h.object_id}")
            print(f"  Outputs: reconstructions/{subj}/<KNEELAND_OUT>/")
        else:
            print(kneeland_production_pipeline_subj.remote(*args_pipe))
    elif step == "roi-corr-audit":
        train_s = os.environ.get("ADAPTER_TRAIN_SUBJ", "subj01")
        print(f"=== ROI correspondence audit (train={train_s} → all subjects) ===")
        print(roi_corr_audit_subj.remote(train_s))
    elif step == "train-diffusion-prior-pooled":
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        rid = run_id or ""
        extra = ["--pooled", "--dual-run-id", dual_rid]
        if os.environ.get("FORCE_PRIOR_CACHE", "0") == "1":
            extra.append("--force-cache")
        print(f"=== Pooled diffusion prior (all subjects) dual={dual_rid} ===")
        if "--detach" in sys.argv or "-d" in sys.argv:
            h = train.spawn("subj01", "train_diffusion_prior.py", epochs, rid, extra)
            print(f"Spawned detached: {h.object_id}")
        else:
            print(train.remote("subj01", "train_diffusion_prior.py", epochs, rid, extra))
    elif step == "train-diffusion-prior-all":
        import re
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        print(f"=== Per-subject diffusion prior for {ALL_SUBJECTS} ===")
        handles = []
        for s in ALL_SUBJECTS:
            s_rid = re.sub(r"subj\d+", s, dual_rid, count=1) if re.search(r"subj\d+", dual_rid) else dual_rid
            extra = ["--dual-run-id", s_rid]
            if os.environ.get("FORCE_PRIOR_CACHE", "0") == "1":
                extra.append("--force-cache")
            handles.append((s, train.spawn(s, "train_diffusion_prior.py", epochs, run_id or "", extra)))
        if "--detach" in sys.argv or "-d" in sys.argv:
            print("Spawned 8 prior training jobs (detached).")
            for s, h in handles:
                print(f"  {s}: {h.object_id}")
        else:
            for s, h in handles:
                print(f"  {s}: {h.get()}")
    elif step == "train-dual-contrastive":
        rid = run_id or ""
        print(f"=== Train 4H_CTR2 dual contrastive {subj} epochs={epochs} ===")
        extra = []
        if os.environ.get("MINDBRIDGE_TEMPERATURE"):
            extra.extend(["--temperature", os.environ["MINDBRIDGE_TEMPERATURE"]])
        if "--detach" in sys.argv or "-d" in sys.argv:
            h = train.spawn(subj, "train_dual_contrastive.py", epochs, rid, extra or None)
            print(f"Spawned detached: {h.object_id}")
        else:
            print(train.remote(subj, "train_dual_contrastive.py", epochs, rid, extra or None))
    elif step == "train-dual-mlp-recon":
        rid = run_id or ""
        extra = []
        for flag, env_key in (
            ("--temperature", "MINDBRIDGE_TEMPERATURE"),
            ("--w-gen-proj", "W_GEN_PROJ"),
            ("--w-gen-vae", "W_GEN_VAE"),
            ("--recon-vae-batch", "RECON_VAE_BATCH"),
        ):
            if os.environ.get(env_key):
                extra.extend([flag, os.environ[env_key]])
        print(f"=== Train dual MLP + recon semantic losses {subj} epochs={epochs} ===")
        if "--detach" in sys.argv or "-d" in sys.argv:
            h = train.spawn(subj, "train_dual_mlp_recon.py", epochs, rid, extra or None)
            print(f"Spawned detached: {h.object_id}")
        else:
            print(train.remote(subj, "train_dual_mlp_recon.py", epochs, rid, extra or None))
    elif step == "train-dual-mlp-e2e":
        rid = run_id or ""
        extra = []
        for flag, env_key in (
            ("--temperature", "MINDBRIDGE_TEMPERATURE"),
            ("--prior-run-id", "PRIOR_RUN_ID"),
            ("--prior-subj", "PRIOR_SUBJ"),
            ("--w-gen-proj", "W_GEN_PROJ"),
            ("--w-gen-vae", "W_GEN_VAE"),
            ("--w-e2e-clip", "W_E2E_CLIP"),
            ("--w-e2e-lpips", "W_E2E_LPIPS"),
            ("--e2e-every", "E2E_EVERY"),
            ("--e2e-max-samples", "E2E_MAX_SAMPLES"),
            ("--e2e-start-frac", "E2E_START_FRAC"),
            ("--ddim-train-steps", "DDIM_TRAIN_STEPS"),
            ("--vd-train-steps", "VD_TRAIN_STEPS"),
            ("--vd-strength", "VD_STRENGTH"),
        ):
            if os.environ.get(env_key):
                extra.extend([flag, os.environ[env_key]])
        print(
            f"=== Train dual MLP + E2E (prior→VD→PNG, CLIP/LPIPS vs GT) {subj} "
            f"epochs={epochs} ==="
        )
        if "--detach" in sys.argv or "-d" in sys.argv:
            h = train.spawn(subj, "train_dual_mlp_e2e.py", epochs, rid, extra or None)
            print(f"Spawned detached: {h.object_id}")
        else:
            print(train.remote(subj, "train_dual_mlp_e2e.py", epochs, rid, extra or None))
    elif step in ("production-perception-subj01", "perception-trials-subj01"):
        dual_rid = run_id or os.environ.get(
            "DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep",
        )
        osd = os.environ.get("PERCEPTION_OUT", "kn_perception_setb_prior_vd03_s29")
        seed = int(os.environ.get("PRIOR_SEED", "29"))
        tau = float(os.environ.get("PRIOR_TEMPERATURE", "1.5"))
        vd = float(os.environ.get("VD_STRENGTH", "0.3"))
        print(
            f"=== Production perception Set-B Kneeland {subj} "
            f"(ridge S1, prior S2 like imagery Set B, no imagery adapter) "
            f"VD={vd} τ={tau} → {osd} ==="
        )
        print(reconstruct_vdvae_dual.remote(
            subj, dual_rid, "4H_CTR2", "best_retrieval", 3, vd, 50, 7.5,
            False, False, "", "prior", "", tau, False, osd,
            "", 0.6, -1.0, 0.85, seed, seed, "set_b", 0.85, -1.0, "", "",
            "vdvae_ridge", "", False, True,
        ))
    elif step in ("recon-pipeline-subj01", "production-kneeland-subj01"):
        dual_rid = run_id or os.environ.get(
            "DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep",
        )
        osd = os.environ.get(
            "KNEELAND_OUT",
            os.environ.get("PIPELINE_OUT", "kn_atxt_a_txt005_b_vd03_s29"),
        )
        print(
            f"=== Production Kneeland A+B subj01 (~65%) → {osd} ==="
        )
        out = recon_pipeline_subj01_modal.remote(
            subj,
            dual_rid,
            osd,
            int(os.environ.get("PRIOR_SEED", "29")),
        )
        print(out)
    elif step == "paper-grids-subj01":
        rid = run_id or os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        osd = os.environ.get(
            "PAPER_OUT",
            os.environ.get("KNEELAND_OUT", "kn_atxt_a_txt005_b_vd03_s29"),
        )
        extra = [
            "--run-id", rid,
            "--beta-adapter-run-id", beta_rid,
            "--out-subdir", osd,
            "--seed", os.environ.get("PRIOR_SEED", "29"),
        ]
        print(f"=== Production Kneeland grids subj01 → {osd} ===")
        if "--detach" in sys.argv or "-d" in sys.argv:
            h = train.spawn("subj01", "paper_grids_subj01.py", 1, rid, extra)
            print(f"Spawned detached: {h.object_id}")
        else:
            print(train.remote("subj01", "paper_grids_subj01.py", 1, rid, extra))
    elif step == "eval-adapter-cross-subject":
        train_subj = os.environ.get("ADAPTER_TRAIN_SUBJ", "subj01")
        test_subjs = os.environ.get("ADAPTER_TEST_SUBJS", "subj02,subj05,subj07")
        beta_rid = beta_adapter_run_id or "20260603_230807_imagery_beta_adapter_subj01"
        dual_rid = os.environ.get("DUAL_RUN_ID", "20260602_053625_train_dual_contrastive_subj01_150ep")
        subset = os.environ.get("IMAGERY_SUBSET", "kneeland")
        extra = [
            "--cross-subject",
            "--adapter-train-subj", train_subj,
            "--test-subjs", test_subjs,
            "--beta-adapter-run-id", beta_rid,
            "--run-id", dual_rid,
            "--subset", subset,
            "--head", "projector",
            "--variant", "4H_CTR2",
        ]
        if os.environ.get("CROSS_PRIOR", "0") == "1":
            extra.append("--cross-prior")
        cross_dec = os.environ.get("CROSS_DECODER", "dual")
        if cross_dec == "joint_ridge":
            extra.extend([
                "--cross-decoder", "joint_ridge",
                "--joint-ridge-run-id",
                os.environ.get("JOINT_RIDGE_RUN_ID", "20260601_053733_joint_ridge_all8"),
            ])
        print(
            f"=== Adapter cross-subject: train={train_subj} → test=[{test_subjs}] "
            f"adapter={beta_rid} decoder={cross_dec} ==="
        )
        if "--detach" in sys.argv or "-d" in sys.argv:
            h = train.spawn(train_subj, "eval_imagery_crossdecode.py", 1, dual_rid, extra)
            print(f"Spawned detached: {h.object_id}")
        else:
            print(train.remote(train_subj, "eval_imagery_crossdecode.py", 1, dual_rid, extra))
    elif step == "cross-decode":
        # Dual contrastive evaluation (4H+CTR2):
        #   AB + image_projector → retrieval in pooled 257-token hidden space
        #   C  + text_projector  → retrieval in CLIP text embedding space
        dual_default_rid = "20260602_053625_train_dual_contrastive_subj01_150ep"
        rid = run_id or os.environ.get("MINDBRIDGE_RUN_ID", dual_default_rid)
        ckpt_prefer_mapped = ckpt_prefer
        if ckpt_prefer == "final":
            ckpt_prefer_mapped = "best_retrieval"
        extra = [
            "--variant", variant,
            "--run-id", rid,
            "--ckpt-prefer", ckpt_prefer_mapped,
            "--set", set,
            "--head", head,
        ]
        print(f"=== Dual cross-decode subj={subj} set={set} head={head} variant={variant} ===")
        print(train.remote(subj, "eval_dual_contrastive_crossdecode.py", 1, rid, extra))
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
    # elif step == "eval-whitening":
    #     extra = ["--split", "both", "--variant", "A"]
    #     if os.environ.get("WHITEN_VARIANT"):
    #         extra = ["--split", os.environ.get("WHITEN_SPLIT", "both"), "--variant", os.environ["WHITEN_VARIANT"]]
    #     print(train.remote(subj, "eval_clip_whitening.py", 1, run_id, extra))
    elif step == "eval-whitening":
        os.environ["MINDBRIDGE_RUN_ID"] = run_id
        print(eval_whitening.remote(subj, run_id, variant))
    elif step == "download-roi":
        print(download_roi_all.remote())
    elif step == "download-imagery-meta":
        print(download_imagery_meta.remote())
    else:
        print(run_pipeline.remote(subj))
