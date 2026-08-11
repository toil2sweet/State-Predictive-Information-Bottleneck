# Local Codex → Git → NSCC → PBS

This repository is the local source of truth for the modified SPIB/HSIC-SPIB
and CTC code. Use the local checkout for Codex edits and version control; use
NSCC only for dependency installation, GPU checks, debugging, and submitted
experiments.

The paths used by the supplied PBS template are:

- local code: `/Users/lidepeng/CodexWorkspace/State-Predictive-Information-Bottleneck`
- NSCC code: `/home/users/nus/depeng/State-Predictive-Information-Bottleneck`
- NSCC conda environment: `/data/projects/11014454/depeng/envs/macil`
- NSCC data/results/logs: `/data/projects/11014454/depeng/`

## 1. Save local changes safely

The current checkout already contains uncommitted user changes. Inspect them
before staging anything:

```bash
cd /Users/lidepeng/CodexWorkspace/State-Predictive-Information-Bottleneck
git status --short
git diff -- SPIB.py SPIB_training.py test_model.py test_model_advanced.py
```

Because `origin` currently points to the upstream repository, do not push your
modified `hsic-spib` branch to `origin`. Create a personal fork/private repository
and add it as a second remote once:

```bash
git remote add personal git@github.com:<your-account>/State-Predictive-Information-Bottleneck.git
```

Stage only source/config/workflow files that belong to the experiment. Do not
use `git add -A` while large trajectories, papers, notebooks, or generated
results are unreviewed:

```bash
git add -p SPIB.py SPIB_training.py test_model.py test_model_advanced.py
git add examples/*.ini hsic_utils.py plot_*.py nscc/ AGENTS.md NSCC_WORKFLOW.md
git commit -m "Extend SPIB with HSIC and NSCC workflow"
git push -u personal hsic-spib
```

For later Codex iterations, use a small commit per logical change. A previous
state can then be inspected or restored without touching the working tree:

```bash
git log --oneline --decorate --all
git show <commit>
git diff <old-commit>..<new-commit>
git restore --source <commit> -- <file>
```

## 2. Prepare the NSCC checkout once

On a login node, create the project directories and clone your personal remote
under `$HOME` (the repository itself is relatively small compared with data and
results):

```bash
mkdir -p /data/projects/11014454/depeng/{data,results,logs}
git clone --branch hsic-spib --single-branch \
  git@github.com:<your-account>/State-Predictive-Information-Bottleneck.git \
  /home/users/nus/depeng/State-Predictive-Information-Bottleneck
cd /home/users/nus/depeng/State-Predictive-Information-Bottleneck
git remote -v
```

If the checkout already exists, configure `personal` there instead of cloning
again. Keep datasets, checkpoints, figures, and logs under the project path.
For a config that refers to project data, use absolute NSCC paths such as
`/data/projects/11014454/depeng/data/...`; paths such as
`double-well_CTC/traj_data.npy` remain relative to the code checkout.

## 3. Update NSCC after each local commit

```bash
cd /home/users/nus/depeng/State-Predictive-Information-Bottleneck
NSCC_GIT_REMOTE=personal NSCC_GIT_BRANCH=hsic-spib nscc/update_code.sh
```

The helper defaults to `personal/hsic-spib`, switches an older `main` checkout
to the `hsic-spib` branch, and then permits only a fast-forward update. It
refuses to pull over tracked server-side edits. Resolve those edits deliberately,
then run the update again; untracked datasets are left untouched.

## 4. Verify the environment in a short GPU interactive job

```bash
qsub -I -q normal \
  -l select=1:ncpus=4:mem=16gb:ngpus=1 \
  -l walltime=00:30:00 -P 11014454

module load miniforge3/24.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /data/projects/11014454/depeng/envs/macil
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")'
```

If the local NSCC configuration uses a different GPU resource name, keep the
same idea but replace `ngpus=1` with the resource spelling shown by NSCC. Do
not run training on the login node.

## 5. Submit a reproducible SPIB/HSIC-SPIB run

From the NSCC checkout:

```bash
qsub -v CONFIG=examples/Four_Well_hsic_config.ini \
  nscc/run_spib_advanced.pbs
```

For the double-well CTC-style configuration:

```bash
qsub -v CONFIG=examples/Double_Well_hsic_config.ini \
  nscc/run_spib_advanced.pbs
```

The template requests one GPU, loads only Miniforge, activates `macil`, prints
the PyTorch/CUDA/GPU information, records the Git commit, and runs
`test_model_advanced.py`. Outputs go to
`/data/projects/11014454/depeng/results/spib/<job-id>/`; logs go to
`/data/projects/11014454/depeng/logs/spib/`.

Check and monitor jobs with:

```bash
qstat -u "$USER"
qstat -f <job-id>
tail -f /data/projects/11014454/depeng/logs/spib/<job-id>.log
```

The code loads trajectory arrays onto the selected CUDA device. For large
trajectories, first use a small subset and confirm H100 memory usage with
`nvidia-smi`; request more CPU memory or a longer walltime only after the test
is stable.

## 6. Recommended iteration loop

1. Modify and smoke-test locally with Codex.
2. Review `git diff`, commit only the logical code/config change, and push to
   `personal`.
3. On NSCC, run `nscc/update_code.sh` and submit a short GPU test.
4. Inspect the log and outputs; then submit the full batch run.
5. Record the commit, config, seed, environment prefix, job ID, and output path.

Codex can review the next change with `/review` before you commit. Keep the
NSCC checkout clean and avoid editing the same files independently on both
machines.
