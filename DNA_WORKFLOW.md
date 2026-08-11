# Local Codex to DNA Cluster workflow

This workflow keeps Codex and Git operations on the local Mac and runs GPU work
through DNA Slurm. The current DNA entry point is
`lidp@zlogin1.ddns.comp.nus.edu.sg`; the older help page still names `zlogin2`.

## Resource choices

- Use `GPUA100` for the default SPIB training run or `GPUA40` when it has a
  shorter queue. Both accept `--gres=gpu:1`.
- One job requests 1 GPU, 8 CPU cores, and 32 GB RAM. This is within the DNA
  limits of 32 CPU cores and 124 GB RAM per allocated GPU.
- Use `sbatch` for formal GPU jobs. Do not run PyTorch training or a long package
  installation on a login node.
- The `GPUR6000` Blackwell node is intentionally not used by these helpers yet.

## One-time DNA checkout

After the local workflow changes have been committed and pushed to the personal
fork, run these commands on DNA:

```bash
git clone --branch hsic-spib --single-branch \
  https://github.com/toil2sweet/State-Predictive-Information-Bottleneck.git \
  /mnt/rna01/lidp/State-Predictive-Information-Bottleneck
cd /mnt/rna01/lidp/State-Predictive-Information-Bottleneck
git remote rename origin personal
mkdir -p /mnt/rna01/lidp/spib-project/{data,envs,logs,results}
```

If the checkout already exists, do not clone over it. Inspect it and use
`dna/update_code.sh` instead.

## Install the SPIB environment

Miniforge is installed at `/mnt/rna01/lidp/miniforge3`. The installation job
uses Mamba and the tracked `dna/spib-environment.yml` specification to create
the isolated Conda environment `/mnt/rna01/lidp/miniforge3/envs/spib`, then
installs pinned PyTorch 2.7.1 with its CUDA 11.8 wheel. The Miniforge package
cache is shared by future environments such as `macil`, but their dependencies
remain isolated:

```bash
cd /mnt/rna01/lidp/State-Predictive-Information-Bottleneck
mkdir -p /mnt/rna01/lidp/spib-project/logs/{slurm,gpu-probe}
sbatch --output=/mnt/rna01/lidp/spib-project/logs/slurm/spib-env-%j.out \
  dna/install_spib_env.slurm
squeue -u lidp
```

After it completes, test the environment on one GPU model at a time:

```bash
sbatch -p GPUA100 \
  --output=/mnt/rna01/lidp/spib-project/logs/gpu-probe/A100-%j.out \
  dna/probe_gpu.slurm
sbatch -p GPUA40 \
  --output=/mnt/rna01/lidp/spib-project/logs/gpu-probe/A40-%j.out \
  dna/probe_gpu.slurm
```

The probe must report `cuda_available=True` and complete a small CUDA tensor
operation before training is submitted. Do not use `conda init` on DNA; for a
manual shell, activate this environment with:

```bash
source /mnt/rna01/lidp/miniforge3/etc/profile.d/conda.sh
conda activate spib
```

## Normal edit-to-run cycle

On the local Mac, make a focused commit on `hsic-spib` and push it:

```bash
git status --short
git add -p
git commit -m "Describe the SPIB experiment change"
git push personal hsic-spib
```

Then the local controller can update DNA and submit a run:

```bash
dna/remote.sh update
dna/remote.sh submit GPUA100 examples/Four_Well_hsic_config.ini
dna/remote.sh status
dna/remote.sh tail JOB_ID
```

Use `GPUA40` in the submit command to choose an A40 instead. The DNA update is
fast-forward only and refuses tracked server-side edits. Submission refuses an
untracked config and records the full Git commit.

Slurm archives that exact commit into `/tmp/lidp/spib/JOB_ID`, copies all config
inputs to node-local storage, trains there, and copies results back only at the
end. Persistent files are written to:

```text
/mnt/rna01/lidp/spib-project/logs/spib/JOB_ID.log
/mnt/rna01/lidp/spib-project/results/spib/JOB_ID-SHORT_COMMIT/
```

The Four Well example inputs are tracked by Git and work directly. Large or
private data must remain outside Git under
`/mnt/rna01/lidp/spib-project/data/`, preserving the relative path used in the
config. For example, `traj_data = [muller/traj_data.npy]` is resolved as
`/mnt/rna01/lidp/spib-project/data/muller/traj_data.npy` when it is not part of
the commit.

Each result directory contains the staged config and a metadata file with the
commit, config, environment packages, Slurm partition/node, and allocated GPU.
