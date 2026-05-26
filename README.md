# mindbridge

Project repository for Stanford [Sherlock](https://www.sherlock.stanford.edu/) HPC.

## GitHub

- **Remote:** `https://github.com/anyampinto/mindbridge.git`
- **SSH (Sherlock):** `git@github.com:anyampinto/mindbridge.git`

## Clone on Sherlock

```bash
cd $GROUP_HOME   # or $SCRATCH / your project directory
git clone git@github.com:anyampinto/mindbridge.git
cd mindbridge
```

If `git clone` over SSH fails, add your Sherlock SSH public key to GitHub:
**Settings → SSH and GPG keys → New SSH key**, then paste `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`).

## Environment setup (Sherlock)

```bash
module load python/3.12.1
python3 -m venv venv
source venv/bin/activate
pip install -U pip
# pip install -r requirements.txt   # when you add dependencies
```

## Sync changes

**From your laptop → GitHub → Sherlock:**

```bash
# laptop (in this repo)
git add -A && git commit -m "your message" && git push

# Sherlock
cd mindbridge && git pull
```

## Slurm job template

Copy and edit `scripts/job.slurm` when you add training or batch jobs.
