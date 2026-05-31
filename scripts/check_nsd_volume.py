"""Verify NSD beta session counts on Modal volume vs public S3 release."""
import subprocess
from pathlib import Path

from download_nsd import ALL_SUBJECTS, max_sessions_for_subject

ROOT = Path("/mnt/mindbridge")
BETA_DIR = "nsddata_betas/ppdata/{subj}/func1pt8mm/betas_fithrf_GLMdenoise_RR"
S3 = "s3://natural-scenes-dataset"


def main() -> None:
    print("NSD perception betas — local volume vs public release:\n")
    all_complete = True
    for subj in ALL_SUBJECTS:
        expected = max_sessions_for_subject(subj)
        p = ROOT / "nsd_data" / BETA_DIR.format(subj=subj)
        have = {f.name for f in p.glob("betas_session*.hdf5")} if p.exists() else set()
        local_n = len(have)
        status = "COMPLETE" if local_n >= expected else "INCOMPLETE"
        if status != "COMPLETE":
            all_complete = False
        print(f"  {subj}: {local_n}/{expected} sessions — {status}")
        if status != "COMPLETE":
            missing = [
                f"betas_session{i:02d}.hdf5"
                for i in range(1, expected + 1)
                if f"betas_session{i:02d}.hdf5" not in have
            ]
            key = f"{S3}/{BETA_DIR.format(subj=subj)}/{missing[0]}"
            r = subprocess.run(
                ["aws", "s3", "ls", key, "--no-sign-request"],
                capture_output=True,
                text=True,
            )
            on_s3 = r.returncode == 0
            print(f"    missing locally: {missing[:3]}{'...' if len(missing) > 3 else ''}")
            print(f"    on public S3: {'yes' if on_s3 else 'no (not in public release)'}")

    print()
    if all_complete:
        print("All subjects have the full public NSD release on the volume.")
    else:
        print("Some files are missing locally AND on S3 — re-run download.")
    print()
    print("Note: subj03/06 have 32 sessions; subj04/08 have 30 — not 40.")
    print("Those later sessions were never collected (scanner/scheduling), not withheld.")


if __name__ == "__main__":
    main()
