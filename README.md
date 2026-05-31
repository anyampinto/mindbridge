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

## Environment setup (uv)

Uses [uv](https://docs.astral.sh/uv/) like your other projects. Creates `.venv/` locally (gitignored); `uv.lock` is committed for reproducible installs.

### Laptop

```bash
cd mindbridge
uv sync
source .venv/bin/activate
# uv add numpy   # add deps as you go
```

### Sherlock

One-time: install uv if it is not already on your PATH:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Then in the repo:

```bash
module load python/3.12.1   # or a version matching requires-python in pyproject.toml
uv sync
source .venv/bin/activate
```

Run without activating: `uv run python your_script.py`

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
