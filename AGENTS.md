# Project workflow

## Source of truth

- Treat the local checkout and all existing uncommitted changes as user-owned.
- Do not delete, reset, or overwrite modified or untracked code, data, notebooks,
  papers, checkpoints, or plots without explicit instruction.
- Keep large trajectories, generated results, and checkpoints outside Git. The
  NSCC project storage is `/data/projects/11014454/depeng/`; the code checkout
  may remain under `/home/users/nus/depeng/`.
- The active experiment branch is `hsic-spib`; NSCC updates must not default to
  upstream `main`.
- The DNA checkout is `/mnt/rna01/lidp/State-Predictive-Information-Bottleneck`;
  persistent data, environments, logs, and results belong under
  `/mnt/rna01/lidp/spib-project/`.
- DNA uses the Miniforge base `/mnt/rna01/lidp/miniforge3`; the default SPIB
  environment is `/mnt/rna01/lidp/miniforge3/envs/spib`. Keep MACIL in a
  separate environment rather than adding its dependencies to SPIB.

## SPIB entry points

- Preliminary run: `python test_model.py ...`
- Config-driven run: `python test_model_advanced.py -config <config.ini>`
- The code automatically uses CUDA when `torch.cuda.is_available()` is true.
- For NSCC batch jobs, set `SPIB_OUTPUT_DIR` and `SPIB_FIG_DIR` so generated
  files do not consume the home quota.

## Change and verification rules

- Make focused changes and preserve the user's HSIC-SPIB/CTC work.
- Before editing, inspect `git status` and the relevant diff.
- Prefer a small smoke test or syntax check before an expensive GPU run.
- Record the Git commit, config file, environment, GPU, random seed, and output
  directory for every reproducible experiment.
- Run GPU work through PBS/NSCC rather than on a login node.
- On DNA, use Slurm `sbatch` and the `GPUA100` or `GPUA40` partition. Never run
  PyTorch training or long environment installation on a DNA login node.
- A DNA run must receive a full Git commit, execute an archived snapshot of that
  commit in `/tmp/$USER`, stage input data there, and copy outputs back after the
  computation.
