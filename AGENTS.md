# Project workflow

## Source of truth

- Treat the local checkout and all existing uncommitted changes as user-owned.
- Do not delete, reset, or overwrite modified or untracked code, data, notebooks,
  papers, checkpoints, or plots without explicit instruction.
- Keep large trajectories, generated results, and checkpoints outside Git.
  A100 uses `/home/project/11004454/depeng`; H100 uses
  `/data/projects/11014454/depeng`. Code mirrors may remain under `$HOME`.
- The active experiment branch is `hsic-spib`; NSCC updates must not default to
  upstream `main`.
- The local Mac checkout is the only editable source of truth. Do not edit the
  same source files independently on NSCC.
- Each NSCC cluster has one runtime mirror at
  `/home/users/nus/depeng/State-Predictive-Information-Bottleneck`. Do not
  create a second mirror on either cluster; reproducibility comes from immutable
  per-job source snapshots under project storage and from Git commits/tags.
- The SPIB environment is `<target-project-root>/envs/spib`.
  Keep MACIL in `/data/projects/11014454/depeng/envs/macil` and do not mix the
  two environments.
- The DNA checkout is `/mnt/rna01/lidp/State-Predictive-Information-Bottleneck`;
  persistent data, environments, logs, and results belong under
  `/mnt/rna01/lidp/spib-project/`.
- DNA uses the Miniforge base `/mnt/rna01/lidp/miniforge3`; the default SPIB
  environment is `/mnt/rna01/lidp/miniforge3/envs/spib`. Keep MACIL in a
  separate environment rather than adding its dependencies to SPIB.
- The HKU CS GPU farm is a temporary extra target, not a replacement for NSCC.
  `/userhome/cs3/lidepeng/TS` is a multi-project workspace; never rsync onto
  that root. The single HKU runtime checkout is
  `/userhome/cs3/lidepeng/TS/State-Predictive-Information-Bottleneck`. Leave
  `State-Predictive-Information-Bottleneck_v1/v2/v3`, `spib`, `ts-dar`, and
  `CTClustering` untouched.
- After `gpu-interactive`, HKU uses the default Conda `base` environment at
  `/userhome/cs3/lidepeng/anaconda3` as the SPIB runtime. Keep `macil` and
  `CL` separate. Do not copy NSCC Conda environments. Code, Four-Well
  trajectories, logs, and results all live under the HKU runtime checkout;
  there is no separate project-storage root.
- Edit source only on the local Mac. Do not edit the same files independently
  on HKU.

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
- Do not stage, commit, tag, or push unless the user explicitly asks to save
  the current code. A request to modify, inspect, test, sync, or run code does
  not authorize a Git publication step.
- The default NSCC target is ASPIRE2A A100 (`11004454`). Use ASPIRE2A+ H100
  (`11014454`) only when the user says H100/ASPIRE2A+ or explicitly passes
  `h100`. Never infer the target from an old job ID.
- Sync the local working tree to the selected NSCC target only when the user asks to test or run it,
  and only if it changed since the previous sync. Use `nscc/sync_code.sh`
  rather than duplicating ad-hoc rsync commands.
- For a request combining local SPIB changes with NSCC execution or monitoring,
  use the repository skill `spib-nscc-runner` and carry the task through job
  completion rather than handing the user the intermediate command sequence.
- When the user explicitly says `hku服务器上执行`, `HKU`, `港大`, or `gpu3`,
  use the repository skill `spib-hku-runner` and the `hku/` helpers. Do not
  change `nscc/` defaults or treat HKU as an `a100`/`h100` target. Sync with
  `hku/sync_code.sh`, enter GPU with `gpu-interactive` via
  `hku/interactive_session.sh` or `hku/run.sh`, activate Conda `base`, and keep
  trajectories, logs, and results under the HKU runtime checkout. After each
  HKU run, if figures exist, download them with `hku/fetch_figures.sh` into
  this repository's `fig/<system>-<job>-<version>-<MMDDTHHMM>/` directory.
  Unspecified execution still defaults to NSCC A100.
- When the user explicitly says `交互模式`, use the reconnectable NSCC
  interactive workflow with a default walltime of 6 hours. Reuse the selected
  target's healthy `spib-a100` or `spib-gpu` allocation, sync local changes before each new execution,
  and keep each run's logs/results under the project directory. When the user
  explicitly says `提交作业` or `batch`, use the immutable batch workflow.
- Keep the interactive allocation after a debug run so it can be reused until
  walltime expiry; cancel it only when the user asks or when continuing would
  waste resources outside the stated debugging session.
- `interactive_session.sh` must automatically initialize the compute shell with
  `interactive_setup.sh`. Verify the `(spib)` prompt; run `spib_gpu_check` only
  when GPU diagnostics are needed. Use manual sourcing only as a recovery
  fallback.
- Short debugging runs may use an uncommitted working snapshot. Meaningful or
  long experiments should normally use a committed and pushed `hsic-spib`
  state through `nscc/submit_version.sh`; reserve annotated tags for milestones.
- Submit batch work through `nscc/submit_job.sh`, which freezes an immutable
  source snapshot before `qsub`. Record the base Git commit, source manifest,
  config, environment, GPU, random seed, job ID, and output directory.
- Run GPU work through PBS/NSCC rather than on a login node.
- On DNA, use Slurm `sbatch` and the `GPUA100` or `GPUA40` partition. Never run
  PyTorch training or long environment installation on a DNA login node.
- A DNA run must receive a full Git commit, execute an archived snapshot of that
  commit in `/tmp/$USER`, stage input data there, and copy outputs back after the
  computation.
- Keep the current development identifier in `VERSION` and accepted checkpoint
  notes in `CHANGELOG.md`. Semantic-version patch numbers do not roll over:
  `1.0.9` is followed by `1.0.10`; use a minor increment for a feature-level
  improvement and a major increment for a stage/method milestone.
- “保存当前版本” permits a reviewed commit and push, but not a tag unless the
  user also asks for a milestone/tag. Use `scripts/prepare_version.sh` to update
  version metadata; never create an automatic commit or tag from that script.
- After completing one user-visible logical code/workflow change, add one short
  `Unreleased` entry with `scripts/log_change.sh`; do not log every failed
  attempt or tiny intermediate edit separately.
- End every user-facing final response in this repository with compact fresh
  target status. Obtain it with `bash nscc/status_summary.sh all` when practical.
  Use `NSCC：无活动作业`, or for active work show mode, job ID, PBS state,
  compute node, and approximate remaining walltime. If authentication or the
  scheduler is unavailable, use `NSCC：状态未知（SSH 不可用）`; never omit the
  line or invent status from stale context. Also append a fresh
  `bash hku/status_summary.sh` line (`HKU-TS：...`) when that helper exists.
