# -*- coding: utf-8 -*-
"""MindBridge wrappers around unmodified brain-diffuser / openai VDVAE code."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _brain_diffuser_root() -> Path:
    env = os.environ.get("BRAIN_DIFFUSER_ROOT")
    if env:
        return Path(env)
    local = Path(__file__).resolve().parent.parent / "vendor" / "brain-diffuser"
    if local.exists():
        return local
    return Path("/root/brain-diffuser")


_BD_ROOT = _brain_diffuser_root()
_VDVAE_DIR = _BD_ROOT / "vdvae"

NUM_LATENT_LAYERS = 31
# Legacy brain-diffuser split table (may not match c*h*w per layer); prefer layer_flat_dims().
LAYER_DIMS = np.array([
    2**4, 2**4,
    2**8, 2**8, 2**8, 2**8,
    2**10, 2**10, 2**10, 2**10, 2**10, 2**10, 2**10, 2**10,
    2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12, 2**12,
    2**12, 2**12,
    2**14,
], dtype=np.int64)


def layer_flat_dims(ref_stats) -> np.ndarray:
    """Per-layer flat dim = c*h*w from a reference VDVAE forward (matches encode hstack)."""
    return np.array(
        [int(np.prod(ref_stats[i]["z"].shape[1:])) for i in range(NUM_LATENT_LAYERS)],
        dtype=np.int64,
    )


def vdvae_model_dir(root: Path) -> Path:
    """VDVAE checkpoints on the MindBridge volume."""
    d = root / "vdvae" / "model"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_vdvae_on_path() -> None:
    vdvae = str(_VDVAE_DIR)
    if vdvae not in sys.path:
        sys.path.insert(0, vdvae)


def default_vdvae_h(root: Path) -> dict:
    model_dir = vdvae_model_dir(root)
    ema = model_dir / "imagenet64-iter-1600000-model-ema.th"
    return {
        "image_size": 64,
        "image_channels": 3,
        "seed": 0,
        "port": 29500,
        "save_dir": "./saved_models/test",
        "data_root": "./",
        "desc": "test",
        "hparam_sets": "imagenet64",
        "restore_path": str(model_dir / "imagenet64-iter-1600000-model.th"),
        "restore_ema_path": str(ema),
        "restore_log_path": str(model_dir / "imagenet64-iter-1600000-log.jsonl"),
        "restore_optimizer_path": str(model_dir / "imagenet64-iter-1600000-opt.th"),
        "dataset": "imagenet64",
        "ema_rate": 0.999,
        "enc_blocks": "64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5",
        "dec_blocks": "1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12",
        "zdim": 16,
        "width": 512,
        "custom_width_str": "",
        "bottleneck_multiple": 0.25,
        "no_bias_above": 64,
        "scale_encblock": False,
        "test_eval": True,
        "warmup_iters": 100,
        "num_mixtures": 10,
        "grad_clip": 220.0,
        "skip_threshold": 380.0,
        "lr": 0.00015,
        "lr_prior": 0.00015,
        "wd": 0.01,
        "wd_prior": 0.0,
        "num_epochs": 10000,
        "n_batch": 4,
        "adam_beta1": 0.9,
        "adam_beta2": 0.9,
        "temperature": 1.0,
        "iters_per_ckpt": 25000,
        "iters_per_print": 1000,
        "iters_per_save": 10000,
        "iters_per_images": 10000,
        "epochs_per_eval": 1,
        "epochs_per_probe": None,
        "epochs_per_eval_save": 1,
        "num_images_visualize": 8,
        "num_variables_visualize": 6,
        "num_temperatures_visualize": 3,
        "mpi_size": 1,
        "local_rank": 0,
        "rank": 0,
        "logdir": "./saved_models/test/log",
    }


class _DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Uint8ImageDataset(Dataset):
    def __init__(self, images: np.ndarray):
        self.images = images.astype(np.uint8)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.fromarray(self.images[idx])
        img = T.functional.resize(img, (64, 64))
        return torch.tensor(np.array(img)).float()


def load_ema_vae(root: Path, device: torch.device | None = None):
    ensure_vdvae_on_path()
    from model_utils import load_vaes, set_up_data  # noqa: WPS433

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = _DotDict(default_vdvae_h(root))
    H.local_rank = 0
    H, preprocess_fn = set_up_data(H)
    ema_vae = load_vaes(H)
    ema_vae.eval()
    if device.type == "cuda":
        ema_vae = ema_vae.to(device)
    return ema_vae, preprocess_fn, H


@torch.no_grad()
def encode_flat_latents(
    ema_vae,
    preprocess_fn,
    images_uint8: np.ndarray,
    *,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> np.ndarray:
    """Encode uint8 HWC images → flat VDVAE latents (N, D), same as brain-diffuser extract."""
    device = device or next(ema_vae.parameters()).device
    loader = DataLoader(Uint8ImageDataset(images_uint8), batch_size=batch_size, shuffle=False)
    chunks: list[np.ndarray] = []
    for batch in loader:
        data_input, _ = preprocess_fn(batch)
        if device.type == "cuda":
            data_input = data_input.to(device)
        activations = ema_vae.encoder.forward(data_input)
        _px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
        batch_parts = [
            stats[i]["z"].detach().cpu().numpy().reshape(len(data_input), -1)
            for i in range(NUM_LATENT_LAYERS)
        ]
        chunks.append(np.hstack(batch_parts).astype(np.float32))
    return np.concatenate(chunks, axis=0)


def flat_to_hier(
    flat: np.ndarray,
    ref_stats,
    layer_dims: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Unflatten latents using per-layer c*h*w sizes + reference spatial shapes."""
    if layer_dims is None:
        layer_dims = layer_flat_dims(ref_stats)
    out: list[np.ndarray] = []
    offset = 0
    for i in range(NUM_LATENT_LAYERS):
        dim = int(layer_dims[i])
        sl = flat[:, offset:offset + dim]
        c, h, w = ref_stats[i]["z"].shape[1:]
        out.append(sl.reshape(len(flat), c, h, w))
        offset += dim
    if offset != flat.shape[1]:
        raise ValueError(
            f"VDVAE flat dim mismatch: layer splits sum to {offset}, flat has {flat.shape[1]}"
        )
    return out


@torch.no_grad()
def decode_flat_latents(
    ema_vae,
    flat: np.ndarray,
    ref_stats,
    *,
    layer_dims: np.ndarray | None = None,
    out_size: int = 512,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> list[Image.Image]:
    """Decode flat VDVAE latents → PIL RGB images (brain-diffuser reconstruct path)."""
    device = device or next(ema_vae.parameters()).device
    hier = flat_to_hier(flat, ref_stats, layer_dims=layer_dims)
    imgs: list[Image.Image] = []
    n = len(flat)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sample = [
            torch.tensor(hier[layer][start:end]).float().to(device)
            for layer in range(NUM_LATENT_LAYERS)
        ]
        px_z = ema_vae.decoder.forward_manual_latents(end - start, sample, t=None)
        samples = ema_vae.decoder.out_net.sample(px_z)
        for arr in samples:
            im = Image.fromarray(arr).resize((out_size, out_size), Image.LANCZOS)
            imgs.append(im)
    return imgs


@torch.no_grad()
def capture_ref_stats(ema_vae, preprocess_fn, device: torch.device):
    """Reference hierarchical latent shapes from a dummy forward pass."""
    x = Uint8ImageDataset(np.zeros((1, 64, 64, 3), dtype=np.uint8))[0].unsqueeze(0)
    data_input, _ = preprocess_fn(x)
    if device.type == "cuda":
        data_input = data_input.to(device)
    activations = ema_vae.encoder.forward(data_input)
    _px_z, stats = ema_vae.decoder.forward(activations, get_latents=True)
    return stats


VDVAE_WEIGHT_URLS = {
    "imagenet64-iter-1600000-model-ema.th": (
        "https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/"
        "imagenet64-iter-1600000-model-ema.th"
    ),
}


def download_vdvae_weights(root: Path, force: bool = False) -> Path:
    """Fetch imagenet64 EMA checkpoint to the MindBridge volume if missing."""
    import urllib.request

    model_dir = vdvae_model_dir(root)
    for name, url in VDVAE_WEIGHT_URLS.items():
        dest = model_dir / name
        if dest.exists() and not force:
            print(f"  VDVAE weights OK: {dest}")
            continue
        print(f"  Downloading {name} ...")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        print(f"  Saved {dest}")
    return model_dir / "imagenet64-iter-1600000-model-ema.th"
