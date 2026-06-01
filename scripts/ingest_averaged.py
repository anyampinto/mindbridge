# -*- coding: utf-8 -*-
"""
Average NSD perception betas and targets to one row per unique image (per subject).

Reads trial-level files from nsd_meta/ (unchanged):
  betas_flat_{subj}.npy, targets_full_{subj}.npz, nsd_expdesign.mat

Writes to nsd_meta/averaged/ (never overwrites trial-level ingestion):
  betas_avg_perception_{subj}.npy
  targets_avg_{subj}.npz
  targets_text_clip.npy + targets_text_image_id_73k.npy  (global, shared)

Usage:
  python ingest_averaged.py --root /mnt/mindbridge
  python ingest_averaged.py --root /mnt/mindbridge --skip-text
  python ingest_averaged.py --root /mnt/mindbridge --text-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
from scipy.io import loadmat

sys.path = [p for p in sys.path if "share/software" not in p and "jupyterlab" not in p]

from paths import averaged_meta_dir, get_root

ALL_SUBJ = [f"subj{i:02d}" for i in range(1, 9)]
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
NSD_STIM_INFO_URL = (
    "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_stim_info_merged.csv"
)


def _print_expdesign_keys(expdesign: dict) -> None:
    keys = sorted(k for k in expdesign if not k.startswith("__"))
    print("\n=== nsd_expdesign.mat keys ===")
    for k in keys:
        arr = expdesign[k]
        shp = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", type(arr))
        print(f"  {k}: shape={shp} dtype={dtype}")


def _find_coco_id_array(expdesign: dict) -> tuple[np.ndarray, str]:
    """Return (n_73k,) COCO image IDs (0-based index = imgBrick row)."""
    candidates = []
    for k in sorted(expdesign.keys()):
        if k.startswith("__"):
            continue
        if "coco" not in k.lower():
            continue
        arr = np.asarray(expdesign[k]).squeeze()
        candidates.append((k, arr))

    print("\n=== COCO-related expdesign fields ===")
    for k, arr in candidates:
        print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")

    for k, arr in candidates:
        if arr.ndim == 1 and arr.size >= 73000:
            ids = arr[:73000].astype(np.int64)
            print(f"  Using expdesign['{k}'] for 73k → COCO mapping")
            return ids, k
        if arr.ndim == 2 and arr.shape[0] >= 8 and arr.shape[1] >= 73000:
            # subject-specific coco ids — use row 0 as canonical image table
            ids = arr[0, :73000].astype(np.int64)
            print(f"  Using expdesign['{k}'][0, :] for 73k → COCO mapping")
            return ids, k

    return np.array([], dtype=np.int64), ""


def _load_coco_id_73k(meta_dir: Path, expdesign: dict) -> np.ndarray:
    """Length-73000 array: coco image id for each imgBrick index (0-based)."""
    ids, src = _find_coco_id_array(expdesign)
    if ids.size >= 73000:
        return ids

    stim_csv = meta_dir / "nsd_stim_info_merged.csv"
    if not stim_csv.exists():
        print(f"  Downloading {stim_csv.name}...")
        urlretrieve(NSD_STIM_INFO_URL, stim_csv)

    import csv

    print(f"  Loading COCO map from {stim_csv}")
    out = np.zeros(73000, dtype=np.int64)
    with open(stim_csv, newline="") as f:
        reader = csv.DictReader(f)
        fields = {c.lower(): c for c in (reader.fieldnames or [])}
        coco_col = (
            fields.get("cocoid")
            or fields.get("coco_id")
            or fields.get("cocoid73k")
        )
        if not coco_col:
            raise KeyError(f"No coco id column in {stim_csv}; columns={reader.fieldnames}")
        idx_col = fields.get("73k_id") or fields.get("nsd_id") or fields.get("index")
        for row_i, row in enumerate(reader):
            try:
                i73 = int(row[idx_col]) if idx_col else row_i
                if 0 <= i73 < 73000:
                    out[i73] = int(float(row[coco_col]))
            except (KeyError, ValueError):
                continue
    print(f"  stim_info CSV: {(out > 0).sum()} non-zero COCO ids")
    return out


def trial_slots_10k(n_trials: int, masterordering: np.ndarray) -> np.ndarray:
    """0-based 10k image slot per trial."""
    mo = np.asarray(masterordering).squeeze()
    return mo[:n_trials].astype(np.int64) - 1


def subject_image_ids_73k(subj_idx: int, subjectim: np.ndarray) -> np.ndarray:
    """(10000,) 0-based imgBrick indices for this subject's 10k images."""
    return np.asarray(subjectim)[subj_idx, :].astype(np.int64) - 1


def average_subject(
    subj: str,
    root: Path,
    out_dir: Path,
    masterordering: np.ndarray,
    subjectim: np.ndarray,
) -> dict:
    subj_idx = int(subj.replace("subj", "")) - 1
    betas_path = root / "nsd_meta" / f"betas_flat_{subj}.npy"
    targets_path = root / "nsd_meta" / f"targets_full_{subj}.npz"

    betas_flat = np.load(betas_path)
    td = np.load(targets_path)
    clip_t = td["clip"].astype(np.float32)
    dino_t = td["dino"].astype(np.float32)
    vae_t = td["vae"].astype(np.float32)

    n_trials = betas_flat.shape[0]
    slots = trial_slots_10k(n_trials, masterordering)
    id_73k_table = subject_image_ids_73k(subj_idx, subjectim)

    unique_slots = np.unique(slots)
    unique_slots.sort()
    n_images = len(unique_slots)

    betas_avg = np.zeros((n_images, betas_flat.shape[1]), dtype=np.float32)
    clip_avg = np.zeros((n_images, clip_t.shape[1]), dtype=np.float32)
    dino_avg = np.zeros((n_images, dino_t.shape[1]), dtype=np.float32)
    vae_avg = np.zeros((n_images, *vae_t.shape[1:]), dtype=np.float32)
    image_id_73k = np.zeros(n_images, dtype=np.int64)
    slot_10k = unique_slots.copy()

    rep_counts = []
    for i, slot in enumerate(unique_slots):
        tidx = np.where(slots == slot)[0]
        rep_counts.append(len(tidx))
        betas_avg[i] = betas_flat[tidx].mean(axis=0)
        clip_avg[i] = clip_t[tidx].mean(axis=0)
        dino_avg[i] = dino_t[tidx].mean(axis=0)
        vae_avg[i] = vae_t[tidx].mean(axis=0)
        image_id_73k[i] = id_73k_table[slot]

    mean_reps = float(np.mean(rep_counts))
    print(
        f"  {subj}: {n_trials} trials → {n_images} unique 10k slots | "
        f"mean reps/image={mean_reps:.2f} (min={min(rep_counts)}, max={max(rep_counts)})"
    )

    np.save(out_dir / f"betas_avg_perception_{subj}.npy", betas_avg)
    np.savez(
        out_dir / f"targets_avg_{subj}.npz",
        clip=clip_avg,
        dino=dino_avg,
        vae=vae_avg,
        slot_10k=slot_10k,
        image_id_73k=image_id_73k,
    )

    return {
        "subj": subj,
        "n_trials": int(n_trials),
        "n_images": int(n_images),
        "mean_reps": float(mean_reps),
    }, image_id_73k


def ensure_coco_captions_json(meta_dir: Path) -> Path:
    """Build coco_captions.json: str(coco_image_id) -> list of caption strings."""
    json_path = meta_dir / "coco_captions.json"
    if json_path.exists():
        return json_path

    zip_path = meta_dir / "annotations_trainval2017.zip"
    captions_json = meta_dir / "captions_train2017.json"

    if not captions_json.exists():
        if not zip_path.exists():
            print(f"  Downloading COCO annotations → {zip_path}")
            urlretrieve(COCO_ANNOTATIONS_URL, zip_path)
        print("  Extracting captions_train2017.json ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("captions_train2017.json"):
                    zf.extract(name, meta_dir)
                    extracted = meta_dir / name
                    extracted.rename(captions_json)
                    break

    print(f"  Building {json_path.name} from COCO annotations ...")
    data = json.loads(captions_json.read_text())
    by_image: dict[str, list[str]] = {}
    for ann in data.get("annotations", []):
        iid = str(int(ann["image_id"]))
        by_image.setdefault(iid, []).append(ann["caption"])

    json_path.write_text(json.dumps(by_image))
    print(f"  Saved {len(by_image)} COCO images with captions")
    return json_path


def compute_clip_text_targets(
    unique_73k: np.ndarray,
    coco_id_73k: np.ndarray,
    meta_dir: Path,
    out_dir: Path,
    device: str = "cuda",
) -> None:
    import torch
    import torch.nn.functional as F
    import open_clip

    captions_path = ensure_coco_captions_json(meta_dir)
    by_image = json.loads(captions_path.read_text())

    n = len(unique_73k)
    clip_text = np.zeros((n, 768), dtype=np.float32)
    missing_coco = 0
    missing_caps = 0

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n=== CLIP text encoding (ViT-L-14, openai) on {dev} ===")
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(dev).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    batch_ids: list[int] = []
    batch_emb: list[np.ndarray] = []

    def flush_batch() -> None:
        if not batch_ids:
            return
        embs = np.stack(batch_emb, axis=0)
        for j, row in enumerate(embs):
            clip_text[batch_ids[j]] = row
        batch_ids.clear()
        batch_emb.clear()

    for i, i73 in enumerate(unique_73k):
        i73 = int(i73)
        if i73 < 0 or i73 >= len(coco_id_73k):
            missing_coco += 1
            continue
        coco_id = int(coco_id_73k[i73])
        if coco_id <= 0:
            missing_coco += 1
            continue
        caps = by_image.get(str(coco_id), [])
        if not caps:
            missing_caps += 1
            continue
        caps = caps[:5]
        tokens = tokenizer(caps).to(dev)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1).mean(dim=0)
            emb = F.normalize(emb, dim=-1)
        batch_ids.append(i)
        batch_emb.append(emb.cpu().numpy().astype(np.float32))

        if len(batch_ids) >= 64:
            flush_batch()
        if (i + 1) % 500 == 0:
            print(f"  encoded {i + 1}/{n}")

    flush_batch()
    print(f"  missing COCO id: {missing_coco}  missing captions: {missing_caps}")

    np.save(out_dir / "targets_text_image_id_73k.npy", unique_73k.astype(np.int64))
    np.save(out_dir / "targets_text_clip.npy", clip_text)

    order = np.argsort(unique_73k)
    np.savez(
        out_dir / "targets_text_clip_index.npz",
        image_id_73k=unique_73k[order],
        clip_text=clip_text[order],
    )

    for subj in ALL_SUBJ:
        tpath = out_dir / f"targets_avg_{subj}.npz"
        if not tpath.exists():
            continue
        td = np.load(tpath)
        i73 = td["image_id_73k"]
        sorted_ids = unique_73k[order]
        sorted_text = clip_text[order]
        pos = np.searchsorted(sorted_ids, i73)
        matched = sorted_ids[pos] == i73
        text_rows = np.zeros((len(i73), 768), dtype=np.float32)
        text_rows[matched] = sorted_text[pos[matched]]
        np.savez(
            tpath,
            clip=td["clip"],
            dino=td["dino"],
            vae=td["vae"],
            slot_10k=td["slot_10k"],
            image_id_73k=td["image_id_73k"],
            clip_text=text_rows,
        )


def run_perception_averaging(root: Path, subjects: list[str]) -> list[dict]:
    meta_dir = root / "nsd_meta"
    out_dir = averaged_meta_dir(root)
    exp_path = meta_dir / "nsd_expdesign.mat"
    assert exp_path.exists(), f"Missing {exp_path}"

    expdesign = loadmat(str(exp_path))
    _print_expdesign_keys(expdesign)
    masterordering = expdesign["masterordering"]
    subjectim = expdesign["subjectim"]

    summaries = []
    id_list = []
    for subj in subjects:
        info, ids_73k = average_subject(subj, root, out_dir, masterordering, subjectim)
        summaries.append(info)
        id_list.append(ids_73k)

    all_73k = np.unique(np.concatenate(id_list))
    all_73k.sort()
    np.save(out_dir / "all_subjects_unique_image_id_73k.npy", all_73k)
    return summaries, all_73k, expdesign


def print_final_summary(out_dir: Path, subjects: list[str]) -> None:
    print("\n=== Saved files (nsd_meta/averaged/) ===")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            if p.suffix == ".npy":
                arr = np.load(p, mmap_mode="r")
                print(f"  {p.name}: shape={arr.shape} dtype={arr.dtype}")
            elif p.suffix == ".npz":
                with np.load(p) as z:
                    parts = ", ".join(f"{k}={z[k].shape}" for k in z.files)
                print(f"  {p.name}: {parts}")

    for subj in subjects:
        b = out_dir / f"betas_avg_perception_{subj}.npy"
        t = out_dir / f"targets_avg_{subj}.npz"
        if b.exists() and t.exists():
            betas = np.load(b, mmap_mode="r")
            with np.load(t) as z:
                ct = z["clip_text"].shape if "clip_text" in z.files else "—"
                print(f"  [{subj}] betas_avg {betas.shape} | clip {z['clip'].shape} | clip_text {ct}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT", "/mnt/mindbridge"))
    parser.add_argument("--subj", default="all", help="subj01 or all")
    parser.add_argument("--skip-text", action="store_true", help="Only average betas/targets (CPU)")
    parser.add_argument("--text-only", action="store_true", help="Only CLIP-text pass (needs prior averaging)")
    parser.add_argument("--device", default="cuda", help="Device for CLIP text encoding")
    args = parser.parse_args()

    root = get_root(args.root)
    meta_dir = root / "nsd_meta"
    out_dir = averaged_meta_dir(root)
    subjects = ALL_SUBJ if args.subj == "all" else [args.subj]

    print(f"Root: {root}")
    print(f"Output dir (separate from trial-level): {out_dir}")

    summaries: list[dict] = []
    all_73k = np.load(out_dir / "all_subjects_unique_image_id_73k.npy") if (
        args.text_only and (out_dir / "all_subjects_unique_image_id_73k.npy").exists()
    ) else None
    expdesign = None

    if not args.text_only:
        result = run_perception_averaging(root, subjects)
        summaries, all_73k, expdesign = result
        manifest = {
            "subjects": summaries,
            "n_global_unique_73k": int(len(all_73k)),
            "out_dir": str(out_dir),
        }
        (out_dir / "ingest_averaged_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )

    if args.skip_text:
        print_final_summary(out_dir, subjects)
        return

    if all_73k is None:
        path = out_dir / "all_subjects_unique_image_id_73k.npy"
        assert path.exists(), "Run without --text-only first to build averaged betas."
        all_73k = np.load(path)

    if expdesign is None:
        expdesign = loadmat(str(meta_dir / "nsd_expdesign.mat"))
        _print_expdesign_keys(expdesign)

    coco_id_73k = _load_coco_id_73k(meta_dir, expdesign)
    compute_clip_text_targets(all_73k, coco_id_73k, meta_dir, out_dir, device=args.device)
    print_final_summary(out_dir, subjects)
    print("\nDone.")


if __name__ == "__main__":
    main()
