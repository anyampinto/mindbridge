#!/bin/bash
#SBATCH --job-name=nsd_download
#SBATCH --output=download_%j.out
#SBATCH --error=download_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=normal

echo "Initializing environment..."
date

# Load the AWS CLI module installed on Sherlock
module load aws-cli/2.15.22

# Go to your target directory
cd /scratch/users/ampinto/mindbridge/
mkdir -p nsd_meta
cd nsd_meta

echo "Downloading nsd_expdesign.mat..."
if command -v aws &> /dev/null; then
    aws s3 cp s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat . --no-sign-request
else
    echo "AWS CLI module load failed, falling back to curl..."
    curl -L -o nsd_expdesign.mat https://natural-scenes-dataset.s3.amazonaws.com/nsddata/experiments/nsd/nsd_expdesign.mat
fi

echo "Downloading nsd_stimuli.hdf5 (~10.5 GB)..."
if command -v aws &> /dev/null; then
    aws s3 cp s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 . --no-sign-request
else
    echo "AWS CLI module load failed, falling back to curl..."
    curl -L -o nsd_stimuli.hdf5 https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5
fi

echo "Verifying files in nsd_meta:"
ls -lh

echo "Job finished!"
date
