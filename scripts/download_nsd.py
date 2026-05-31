"""Download NSD perception fMRI, stimulus images, and NSD-Imagery betas via AWS CLI.

Uses the public NSD S3 bucket documented in the NSD Data Manual:
- https://cvnlab.slite.page/p/CT9Fwl4_hc/NSD-Data-Manual
- https://cvnlab.slite.page/p/IB6BSeW_7o (Data Access Agreement)
- https://registry.opendata.aws/nsd/

Before downloading, complete the NSD Data Access form:
https://naturalscenesdataset.org/

AWS examples (public bucket, no account required):
  aws s3 ls s3://natural-scenes-dataset --no-sign-request
  aws s3 cp s3://natural-scenes-dataset/<key> /local/path --no-sign-request
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from paths import get_root

S3_BUCKET = "s3://natural-scenes-dataset"
AWS_REGION = "us-east-2"
BETA_VERSION = "betas_fithrf_GLMdenoise_RR"
IMAGERY_BETA_VERSION = "nsdimagerybetas_fithrf_GLMdenoise_RR"
ALL_SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]

# Public S3 release: four subjects completed 40 sessions; others had fewer scans.
NSD_MAX_SESSIONS: dict[str, int] = {
    "subj01": 40,
    "subj02": 40,
    "subj03": 32,
    "subj04": 30,
    "subj05": 40,
    "subj06": 32,
    "subj07": 40,
    "subj08": 30,
}


def max_sessions_for_subject(subj: str) -> int:
    try:
        return NSD_MAX_SESSIONS[subj]
    except KeyError as exc:
        raise ValueError(f"Unknown subject {subj!r}") from exc


def _aws_base_cmd() -> list[str]:
    return [
        "aws",
        "--region",
        AWS_REGION,
        "--cli-read-timeout",
        "0",
        "--cli-connect-timeout",
        "60",
    ]


def require_aws_cli() -> None:
    if shutil.which("aws") is None:
        raise RuntimeError(
            "AWS CLI not found. Install it first:\n"
            "  brew install awscli\n"
            "  # or: pip install awscli\n"
            "  # Modal image installs it automatically via apt."
        )


def aws_s3_cp(s3_uri: str, dest: Path, *, force: bool = False) -> Path:
    """Copy one S3 object to a local file."""
    require_aws_cli()
    dest = Path(dest)
    if dest.exists() and not force:
        print(f"  exists: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_aws_base_cmd(),
        "s3",
        "cp",
        s3_uri,
        str(dest),
        "--no-sign-request",
    ]
    if os.environ.get("NSD_AWS_DRYRUN", "0") == "1":
        cmd.append("--dryrun")
    print(f"  aws cp {s3_uri} -> {dest}")
    subprocess.run(cmd, check=True)
    return dest


def aws_s3_sync(
    s3_prefix: str,
    dest_dir: Path,
    *,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
    force: bool = False,
) -> Path:
    """Sync an S3 prefix to a local directory with optional include/exclude globs."""
    require_aws_cli()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        *_aws_base_cmd(),
        "s3",
        "sync",
        s3_prefix.rstrip("/") + "/",
        str(dest_dir),
        "--no-sign-request",
    ]
    if not force:
        cmd.extend(["--size-only"])
    for pattern in excludes or ["*"]:
        cmd.extend(["--exclude", pattern])
    for pattern in includes or []:
        cmd.extend(["--include", pattern])
    if os.environ.get("NSD_AWS_DRYRUN", "0") == "1":
        cmd.append("--dryrun")

    print(f"  aws sync {s3_prefix} -> {dest_dir}")
    subprocess.run(cmd, check=True)
    return dest_dir


def local_path(root: Path, rel_key: str) -> Path:
    return get_root(root) / "nsd_data" / rel_key


def meta_path(root: Path, name: str) -> Path:
    return get_root(root) / "nsd_meta" / name


def perception_beta_key(subj: str, session: int) -> str:
    sess = f"session{session:02d}"
    return (
        f"nsddata_betas/ppdata/{subj}/func1pt8mm/"
        f"{BETA_VERSION}/betas_{sess}.hdf5"
    )


def perception_beta_path(root: Path, subj: str, session: int) -> Path:
    return local_path(root, perception_beta_key(subj, session))


def imagery_beta_path(root: Path, subj: str) -> Path:
    rel = (
        f"nsddata_betas/ppdata/{subj}/func1pt8mm/"
        f"{IMAGERY_BETA_VERSION}/betas_nsdimagery.hdf5"
    )
    return local_path(root, rel)


def download_shared_metadata(root: Path, *, stimuli: bool = True, force: bool = False) -> None:
    root = get_root(root)
    print("\n=== NSD shared metadata (AWS CLI) ===")

    aws_s3_cp(
        f"{S3_BUCKET}/nsddata/experiments/nsd/nsd_expdesign.mat",
        meta_path(root, "nsd_expdesign.mat"),
        force=force,
    )

    if stimuli:
        aws_s3_cp(
            f"{S3_BUCKET}/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5",
            meta_path(root, "nsd_stimuli.hdf5"),
            force=force,
        )


def download_imagery_metadata(root: Path, *, force: bool = False) -> None:
    root = get_root(root)
    print("\n=== NSD-Imagery experiment metadata (AWS CLI) ===")
    aws_s3_sync(
        f"{S3_BUCKET}/nsddata/experiments/nsdimagery",
        root / "nsd_meta" / "nsdimagery",
        excludes=["*"],
        includes=["*.mat"],
        force=force,
    )


def download_perception_sessions(
    subj: str,
    root: Path,
    *,
    sessions: range | list[int] | None = None,
    force: bool = False,
) -> None:
    root = get_root(root)
    if sessions is None:
        sessions = range(1, 41)

    print(f"\n=== NSD perception betas ({subj}, {len(list(sessions))} sessions) ===")

    aws_s3_cp(
        f"{S3_BUCKET}/nsddata/ppdata/{subj}/func1pt8mm/roi/nsdgeneral.nii.gz",
        local_path(root, f"nsddata/ppdata/{subj}/func1pt8mm/roi/nsdgeneral.nii.gz"),
        force=force,
    )

    includes = [f"betas_session{session:02d}.hdf5" for session in sessions]
    aws_s3_sync(
        f"{S3_BUCKET}/nsddata_betas/ppdata/{subj}/func1pt8mm/{BETA_VERSION}",
        local_path(root, f"nsddata_betas/ppdata/{subj}/func1pt8mm/{BETA_VERSION}"),
        excludes=["*"],
        includes=includes,
        force=force,
    )


def download_imagery_betas(subj: str, root: Path, *, force: bool = False) -> None:
    root = get_root(root)
    print(f"\n=== NSD-Imagery betas ({subj}) ===")
    aws_s3_cp(
        f"{S3_BUCKET}/nsddata_betas/ppdata/{subj}/func1pt8mm/"
        f"{IMAGERY_BETA_VERSION}/betas_nsdimagery.hdf5",
        imagery_beta_path(root, subj),
        force=force,
    )


def ensure_nsd_data(
    subj: str,
    root: Path | None = None,
    *,
    sessions: range | list[int] | None = None,
    download_stimuli: bool = True,
    download_imagery: bool = True,
    force: bool = False,
) -> None:
    """Download NSD perception data, COCO stimuli, and NSD-Imagery via AWS CLI."""
    root = get_root(root)

    if os.environ.get("NSD_SKIP_DOWNLOAD", "0") == "1":
        print("NSD_SKIP_DOWNLOAD=1 — skipping remote downloads.")
        return

    require_aws_cli()
    download_shared_metadata(root, stimuli=download_stimuli, force=force)
    subj_sessions = (
        parse_sessions(sessions, subj=subj)
        if isinstance(sessions, str)
        else (sessions or parse_sessions("all", subj=subj))
    )
    download_perception_sessions(subj, root, sessions=subj_sessions, force=force)

    if download_imagery:
        download_imagery_metadata(root, force=force)
        download_imagery_betas(subj, root, force=force)

    print("\nNSD AWS download step complete.")


def ensure_all_nsd_data(
    root: Path | None = None,
    *,
    subjects: list[str] | None = None,
    sessions: range | list[int] | None = None,
    download_stimuli: bool = True,
    download_imagery: bool = True,
    force: bool = False,
) -> None:
    """Download full NSD + NSD-Imagery for all subjects (shared files once)."""
    root = get_root(root)
    subjects = subjects or ALL_SUBJECTS

    if os.environ.get("NSD_SKIP_DOWNLOAD", "0") == "1":
        print("NSD_SKIP_DOWNLOAD=1 — skipping remote downloads.")
        return

    require_aws_cli()
    print(f"\n{'=' * 60}")
    print(f"Full NSD download: {len(subjects)} subjects, sessions={sessions or 'all'}")
    print(f"Root: {root}")
    print(f"{'=' * 60}")

    download_shared_metadata(root, stimuli=download_stimuli, force=force)
    if download_imagery:
        download_imagery_metadata(root, force=force)

    for idx, subj in enumerate(subjects, start=1):
        print(f"\n>>> Subject {idx}/{len(subjects)}: {subj}")
        subj_sessions = (
            parse_sessions(sessions, subj=subj)
            if isinstance(sessions, str)
            else (sessions or parse_sessions("all", subj=subj))
        )
        download_perception_sessions(subj, root, sessions=subj_sessions, force=force)
        if download_imagery:
            download_imagery_betas(subj, root, force=force)

    print(f"\nFull NSD download complete for: {', '.join(subjects)}")


def parse_sessions(spec: str, *, subj: str | None = None) -> list[int]:
    """Parse '1-40', '1,2,3', or 'all' into a session list."""
    spec = spec.strip().lower()
    if spec in {"all", "*", ""}:
        upper = max_sessions_for_subject(subj) if subj else 40
        return list(range(1, upper + 1))
    sessions: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            sessions.extend(range(int(start), int(end) + 1))
        else:
            sessions.append(int(part))
    return sessions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NSD + NSD-Imagery from s3://natural-scenes-dataset via AWS CLI"
    )
    parser.add_argument("--subj", default=os.environ.get("NSD_SUBJ", "subj01"))
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Download all 8 NSD subjects (subj01-subj08)",
    )
    parser.add_argument("--root", default=os.environ.get("MINDBRIDGE_ROOT"))
    parser.add_argument("--sessions", default=os.environ.get("NSD_SESSIONS", "all"))
    parser.add_argument("--no-stimuli", action="store_true")
    parser.add_argument("--no-imagery", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Pass --dryrun to aws s3 cp/sync (preview only)",
    )
    args = parser.parse_args()

    if args.dryrun:
        os.environ["NSD_AWS_DRYRUN"] = "1"

    session_spec = args.sessions
    if args.all_subjects or os.environ.get("NSD_ALL_SUBJECTS", "0") == "1":
        ensure_all_nsd_data(
            root=get_root(args.root),
            sessions=session_spec,
            download_stimuli=not args.no_stimuli,
            download_imagery=not args.no_imagery,
            force=args.force,
        )
    else:
        ensure_nsd_data(
            subj=args.subj,
            root=get_root(args.root),
            sessions=session_spec,
            download_stimuli=not args.no_stimuli,
            download_imagery=not args.no_imagery,
            force=args.force,
        )
