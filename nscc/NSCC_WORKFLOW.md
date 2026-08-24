# NSCC workflow: local Codex → ASPIRE2A A100/H100

The Mac checkout is the only editable source. Each physically separate NSCC
cluster has one synchronized runtime mirror plus immutable per-job snapshots.
Commands default to ASPIRE2A A100; append `h100` to select ASPIRE2A+ H100.

## Ask Codex directly

Normal requests can be one sentence:

> 修改 SPIB 的 `<需求>`，使用 `<config.ini>` 在 NSCC 交互模式执行并查看结果。

or:

> 修改 SPIB 的 `<需求>`，使用 `<config.ini>` 提交作业，持续查看日志并分析结果。

Codex inspects and edits locally, checks the change, synchronizes when needed,
runs through PBS, monitors the job, and reports logs/results. These requests do
not save a Git version unless they explicitly include “保存当前版本”.

## Storage model

| Purpose | A100 (default) | H100 |
| --- | --- | --- |
| Cluster / GPU | ASPIRE2A / A100 40 GB | ASPIRE2A+ / H100 80 GB |
| Project | `11004454` | `11014454` |
| Project root | `/home/project/11004454/depeng` | `/data/projects/11014454/depeng` |
| Conda environment | `<project>/envs/spib` | `<project>/envs/spib` |
| Logs/results/snapshots | `<project>/{logs,results,run_sources}/spib` | same layout |

Editable source remains
`/Users/lidepeng/CodexWorkspace/State-Predictive-Information-Bottleneck`;
the runtime mirror on both clusters is
`/home/users/nus/depeng/State-Predictive-Information-Bottleneck`.
On ASPIRE2A, `/home/project/11004454` and `/data/projects/11004454` resolve to
the same project directory; the workflow uses the documented `/home/project`
spelling and loads `miniforge3/25.3.1` explicitly.

A previous secondary server checkout was retired after successful interactive
and batch validation. Its recoverable archive is under
`/data/projects/11014454/depeng/legacy_code_backups/`; do not create another
runtime checkout. Per-job snapshots remain separate because they prevent a
queued run from seeing later code changes.

## Daily high-frequency loop

```text
Codex edits local files
        ↓
local syntax/smoke checks
        ↓
run requested?
  no  → stop locally
  yes → sync_code.sh → one NSCC runtime mirror
                           ↓
              interactive or immutable batch
                           ↓
                    logs + results
```

Synchronize only before the next NSCC execution when local files changed:

```bash
bash nscc/sync_code.sh          # A100
bash nscc/sync_code.sh h100     # H100
```

`sync_code.sh` includes tracked files and new source/config/workflow files, but
omits papers, datasets, checkpoints, generated figures/results, and unrelated
projects. It verifies the remote mirror after transfer.

### Interactive mode

Use for frequent short debugging runs. The default allocation is one GPU,
4 CPUs, 16 GB, and six hours. It is kept in NSCC tmux and can be reattached.

```bash
bash nscc/interactive_session.sh start          # A100
bash nscc/interactive_session.sh status
bash nscc/interactive_session.sh attach

bash nscc/interactive_session.sh start h100     # H100
bash nscc/interactive_session.sh status h100
bash nscc/interactive_session.sh attach h100
```

Once the allocation reaches a compute node:

`interactive_session.sh` automatically changes to the runtime checkout, loads
Miniforge, activates the selected target's `<project>/envs/spib`, and defines
`spib_run`. It does not import PyTorch or probe the GPU during startup, so a
ready prompt appears quickly. Run `spib_gpu_check` only when GPU diagnostics
are needed. Attaching normally opens directly at:

```text
(spib) depeng@a2ap-dgx...:~/State-Predictive-Information-Bottleneck$
```

Then run `spib_run examples/Four_Well_hsic_config.ini`. If automatic setup ever
reports `failed`, recover manually with `source nscc/interactive_setup.sh` from
the runtime checkout.

After another local edit, wait for the current Python process to finish, run
`bash nscc/sync_code.sh` on the Mac, then call `spib_run` again in the same
allocation. Each run has its own project-backed log/result directory and records
the code version, base commit, config, GPU, and exit status. Since an
interactive shell runs on NSCC and cannot write to the Mac directly, retrieve
its figures after the run from a Mac terminal with
`bash nscc/fetch_figures.sh <run-id>`. The helper reads the recorded code
version and stores images under `fig/<job-id>-<version>/`, for example
`fig/203736.pbs111-0.3.0/`.

### Batch mode with the current working tree

Use for disconnect-safe runs and queued work, including uncommitted debugging:

```bash
bash nscc/submit_working.sh examples/Four_Well_hsic_config.ini          # A100
bash nscc/watch_job.sh <job-id>

bash nscc/submit_working.sh examples/Four_Well_hsic_config.ini h100     # H100
bash nscc/watch_job.sh <job-id> h100
```

The submission helper synchronizes once, freezes an immutable source snapshot,
submits it, and prints the job ID and paths. A later synchronization cannot
change that queued or running job. When `watch_job.sh` reaches a terminal state,
it automatically copies that run's remote `fig/` directory to
`/Users/lidepeng/CodexWorkspace/State-Predictive-Information-Bottleneck/fig/<target>/<job-id>-<version>/`.
Only the figure subtree is retrieved; trajectories, checkpoints, and other
large result data remain on NSCC. Set `SPIB_AUTO_FETCH_FIGURES=0` only when a
one-off batch run should skip this automatic retrieval.

## Versions and recovery

Git commits are recoverable checkpoints. Annotated tags are reserved for
stage-level research milestones. `VERSION` identifies the current code line;
`CHANGELOG.md` summarizes accepted checkpoints.

After one meaningful local change succeeds, record it without saving a version:

```bash
bash scripts/log_change.sh "Improve adaptive HSIC bandwidth handling"
```

This only adds an `Unreleased` bullet. It does not change `VERSION` or perform
any Git action; tiny intermediate edits and failed attempts are grouped into
the final logical change.

Semantic version rules:

- patch: fixes/refinements, such as `1.0.9` → `1.0.10`;
- minor: backward-compatible feature-level improvement, such as `1.0.10` → `1.1.0`;
- major: method/stage milestone, such as `1.1.0` → `2.0.0`.

There is no decimal rollover from `1.0.9` to `1.1.0`.

When the user says “保存当前版本为 1.0.1”, Codex should:

1. inspect the complete relevant diff and exclude data/papers/results;
2. run `bash scripts/prepare_version.sh 1.0.1 "<summary>"`;
3. review `VERSION`, `CHANGELOG.md`, source, config, and workflow changes;
4. selectively commit and push `personal/hsic-spib`;
5. not create a tag unless the user also requests a milestone/tag.

When the user says “这是阶段性提升，保存为 2.0.0 并增加 tag”, Codex performs
the same checks, then creates and pushes an annotated
`HSIC-SPIB_V2.0.0` tag after the commit succeeds. If the user supplies another
exact tag name such as `HSIC_V2.0.0`, use that requested name instead.

Run an exact saved checkpoint rather than the current working tree with:

```bash
bash nscc/submit_version.sh <commit-or-tag> examples/Four_Well_hsic_config.ini
bash nscc/watch_job.sh <job-id>
```

This fetches and archives the exact pushed commit. Uncommitted runtime files are
excluded. Existing historical tags such as `v0.1.0-hsic-spib` and
`v0.2.0-hsic-spib-ctc` remain valid historical milestones.

To recover, first inspect `git log`, `git show <ref>`, and `CHANGELOG.md`.
Prefer a new branch or selective file restoration after explicit confirmation;
do not destructively reset over user-owned local changes.

## Logs, results, status, cancellation

```bash
bash nscc/watch_job.sh <job-id>
bash nscc/status_summary.sh          # A100
bash nscc/status_summary.sh h100
bash nscc/status_summary.sh all
ssh nscc 'qstat -f <job-id>'
ssh nscc 'qdel <job-id>'
```

Every batch result directory retains the submitted config and PBS script,
version, commit, source mode, source manifest, and working-tree status/patch
where applicable. Completion requires both `exit_status=0` and expected output
files—not merely a returned job ID.

## Helpers

| Script | Responsibility |
| --- | --- |
| `nscc/sync_code.sh` | Mirror the local working tree to the one NSCC runtime checkout |
| `nscc/target_config.sh` | Central A100/H100 host, project, module, and resource defaults |
| `nscc/bootstrap_target.sh` | Create target project directories and initialize its runtime mirror |
| `nscc/install_spib_env.sh` | Submit the one-time target-specific SPIB environment build |
| `nscc/probe_target.sh` | Submit a five-minute target-specific GPU/environment probe |
| `nscc/sync_data.sh` | Copy persistent input data between H100 and A100 project storage |
| `nscc/submit_working.sh` | Sync, freeze, and submit the current working tree |
| `nscc/submit_version.sh` | Submit an exact pushed commit or tag |
| `nscc/submit_job.sh` | Server-side snapshot creation and `qsub` |
| `nscc/watch_job.sh` | Follow PBS/logs/results and automatically retrieve batch figures |
| `nscc/fetch_figures.sh` | Retrieve one batch or interactive run's remote `fig/` directory |
| `nscc/interactive_session.sh` | Start/reuse/attach the six-hour GPU allocation |
| `nscc/interactive_auto_init.sh` | Activate SPIB automatically when the compute prompt is ready |
| `nscc/interactive_setup.sh` | Activate SPIB and define `spib_run` |
| `nscc/status_summary.sh` | Print a compact live NSCC allocation summary |
| `nscc/probe_gpu.pbs` | Verify PBS, the SPIB environment, and CUDA without training |
| `scripts/log_change.sh` | Record one logical unreleased change without Git actions |
| `scripts/prepare_version.sh` | Update version metadata without Git publication |

Unattended Codex execution requires password-free `ssh nscc`. If the key is not
accepted, install the existing local public key from your own terminal:

```bash
ssh-copy-id -o BatchMode=no -i ~/.ssh/id_ed25519.pub nscc
ssh -o BatchMode=yes nscc true
```

ASPIRE2A currently advertises password/GSSAPI authentication but not public-key
authentication. Its `nscc-a100` SSH alias therefore reuses a 12-hour
ControlMaster connection. Start that master once from a local terminal with
`ssh -MNf -o BatchMode=no nscc-a100`; subsequent workflow commands reuse it.
