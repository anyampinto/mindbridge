#!/usr/bin/env bash
# Download NSD data via AWS CLI (public bucket, no AWS account required).
# Complete the NSD Data Access Agreement first:
# https://cvnlab.slite.page/p/IB6BSeW_7o
#
# Usage:
#   export MINDBRIDGE_ROOT=/path/to/mindbridge
#   export NSD_SUBJ=subj01
#   bash scripts/download_nsd_aws.sh
#   bash scripts/download_nsd_aws.sh --dryrun
#   bash scripts/download_nsd_aws.sh --sessions 1-3 --no-stimuli

set -euo pipefail

ROOT="${MINDBRIDGE_ROOT:-/mnt/mindbridge}"
SUBJ="${NSD_SUBJ:-subj01}"
SESSIONS="${NSD_SESSIONS:-all}"
DRYRUN="${NSD_AWS_DRYRUN:-0}"
DOWNLOAD_STIMULI=1
DOWNLOAD_IMAGERY=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dryrun) DRYRUN=1; shift ;;
    --sessions) SESSIONS="$2"; shift 2 ;;
    --no-stimuli) DOWNLOAD_STIMULI=0; shift ;;
    --no-imagery) DOWNLOAD_IMAGERY=0; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

export MINDBRIDGE_ROOT="$ROOT"
export NSD_SUBJ="$SUBJ"
export NSD_SESSIONS="$SESSIONS"
export NSD_AWS_DRYRUN="$DRYRUN"
export NSD_DOWNLOAD_STIMULI="$DOWNLOAD_STIMULI"
export NSD_DOWNLOAD_IMAGERY="$DOWNLOAD_IMAGERY"

python "$(dirname "$0")/download_nsd.py"
