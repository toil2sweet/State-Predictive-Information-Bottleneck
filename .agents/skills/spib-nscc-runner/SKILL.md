---
name: spib-nscc-runner
description: End-to-end local HSIC-SPIB/CTC modification and NUS NSCC PBS GPU execution on ASPIRE2A A100 or ASPIRE2A+ H100, including synchronization, immutable snapshots, interactive debugging, live logs, results, and versions. Use for SPIB code/jobs on either NSCC cluster. Do not use for DNA Slurm or unrelated repositories.
---

# SPIB NSCC Runner

Treat one natural-language request as one end-to-end operation. Do not make the
user reproduce the manual runbook when the repository helpers can perform it.

## Fixed context

- Use the repository containing this skill as the local editable source.
- Require the local branch `hsic-spib`.
- Use `examples/Four_Well_hsic_config.ini` as the smoke-test config only when
  the user does not name a config and the changed code does not imply another.
- Default to ASPIRE2A A100, Project `11004454`. Select ASPIRE2A+ H100,
  Project `11014454`, only when the user explicitly says H100/ASPIRE2A+.
- Use `<selected-project-root>/envs/spib`; never substitute `macil` or copy a
  Conda environment between clusters.
- Use PBS GPU jobs on NSCC. Never train on a login node.
- Preserve all pre-existing tracked and untracked user files.
- Never stage, commit, tag, push, reset, or restore unless the user explicitly
  asks to save or recover a version.
- Edit source only on the local Mac. Never make an independent server-side edit.
- Use only one runtime mirror per physical cluster at
  `/home/users/nus/depeng/State-Predictive-Information-Bottleneck`. Do not
  create or depend on a second runtime checkout on either cluster.

## Interpret the request

- “修改代码” means edit and verify locally only.
- “交互模式修改/执行/调试” selects the reconnectable interactive workflow
  below, with a default six-hour allocation.
- “提交作业执行”, “batch”, or “批量作业” selects the immutable working batch
  workflow below.
- A generic “修改后在 NSCC 执行/测试并看日志” without an explicit mode uses
  batch because it survives local disconnects and preserves immutable source.
- “只同步” means run `bash nscc/sync_code.sh [target]` and stop after verification.
- “查看 job/log/result” means inspect the existing job without syncing code.
- “交互调试” explicitly selects the interactive workflow.
- “正式实验” or a long scientifically meaningful run should normally use a
  committed stable state. If the user has not explicitly authorized saving,
  ask before committing; do not infer publication authority.
- “保存当前版本” authorizes a reviewed checkpoint: update `VERSION` and
  `CHANGELOG.md`, selectively commit, and push `hsic-spib`. It does not
  authorize a tag.
- “阶段性提升”, “里程碑”, or “增加 tag” authorizes an annotated milestone
  tag after checks pass. Patch versions have no decimal rollover (`1.0.9` →
  `1.0.10`); use minor for feature-level improvements and major for a
  stage/method milestone.

## Default modify-and-run workflow

1. Inspect before editing:
   - run `git status -sb`;
   - inspect relevant existing diffs and implementation;
   - identify the entry point, config, data references, and expected outputs.
2. Make the smallest local change that satisfies the request. Keep algorithm,
   configuration, PBS, and plotting changes focused.
3. Run proportionate local checks before consuming a GPU:
   - run `python -m py_compile` for changed Python files;
   - run targeted CPU/unit checks when dependencies and data allow;
   - run `bash -n` for changed shell or PBS files;
   - inspect `git diff --check` and the final relevant diff.
4. After one user-visible logical change passes its local checks, record one
   concise `Unreleased` entry with `bash scripts/log_change.sh "<summary>"`.
   Group tiny intermediate edits and failed attempts into that final entry.
5. Resolve the target using `nscc/target_config.sh`; omitted target is `a100`.
   Verify non-interactive SSH to that target before submission. A100 uses the
   `nscc-a100` alias; H100 uses the `nscc` alias.

   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=10 <selected-host> true
   ```

   Continue automatically when it succeeds. If it fails, preserve the local
   changes and ask the user to complete one-time SSH key/agent authentication.
   When `/Users/lidepeng/.ssh/id_ed25519.pub` exists, give this exact manual
   If A100 advertises only password/GSSAPI and omits `publickey`, do not retry
   `ssh-copy-id`. Ask the user to start its configured persistent connection
   once with `ssh -MNf -o BatchMode=no nscc-a100`; Codex can reuse that
   ControlMaster for 12 hours. The user must enter the NSCC password in their
   own terminal. Do not ask for a password or leave a hidden prompt running.
6. Submit the current uncommitted working tree from the repository root:

   ```bash
   bash nscc/submit_working.sh <config.ini> [a100|h100]
   ```

   This helper synchronizes the local tree, freezes an immutable working snapshot,
   submits one GPU job, and prints provenance. Do not replace it with ad-hoc
   `rsync`, direct execution on the login node, or a bare `qsub`.
7. Parse the exact `job=<job-id>` line. Report the job ID, config, snapshot,
   log path, and result path immediately.
8. Follow the same job until it reaches a terminal state:

   ```bash
   bash nscc/watch_job.sh <job-id> [a100|h100]
   ```

   Start it as an ongoing shell session. Poll new output every 20–45 seconds,
   and give concise commentary updates at queue-state changes or meaningful log
   milestones. Do not leave the user without an update for more than 60 seconds.
   A queued `Q` job with no log yet is normal; keep waiting.
9. At completion, inspect the recorded exit status, traceback or final metrics,
   result file listing, submitted config, and source provenance. Retrieve and
   visually inspect relevant figures only when they help answer the request.

Do not claim success merely because `qsub` returned a job ID. Success requires
the PBS job to finish with `exit_status=0` and the expected outputs to exist.

## Failure handling

- Classify failures as code, configuration, data, environment, scheduler,
  resource, or scientific-result problems.
- For a clear source bug within the requested scope that fails quickly, make one
  focused local fix, rerun local checks, resubmit a new working snapshot, and watch
  the new job automatically. Preserve both job IDs.
- Do not silently change scientifically meaningful hyperparameters, datasets,
  random seeds, requested resources, or algorithm semantics to force success.
- Do not install packages, rebuild the Conda environment, delete snapshots, or
  cancel jobs unless the user requested that action or it is already explicitly
  in scope.
- If the same cause remains after one automatic fix/rerun, stop and report the
  evidence rather than looping or consuming more GPU time.
- If SSH requires a password/key interaction that tools cannot satisfy, pause
  and ask the user to authenticate. Never request or expose a password/token in
  logs or commands.

## First use of a target

Treat each cluster as a separate machine. Do not copy Conda environments across
clusters. After password-free SSH is available, initialize a new target in this
order:

```bash
bash nscc/bootstrap_target.sh a100
bash nscc/sync_data.sh h100 a100
bash nscc/install_spib_env.sh a100
bash nscc/probe_target.sh a100
```

Monitor the returned install/probe job IDs with `watch_job.sh <job> a100`.
Confirm the actual project mount and Miniforge module from bootstrap output
before accepting defaults. Data synchronization transfers only persistent
`data/`; code is synchronized independently and environments/logs/results are
never copied. Finish with one Four-Well batch smoke test and require
`exit_status=0` plus expected outputs.

## Interactive debugging

Use this workflow when the user explicitly requests `交互模式`, or when a batch
traceback is insufficient and terminal inspection is necessary. The allocation
defaults to six hours and is intentionally reusable for frequent short runs.

1. Inspect and check local changes exactly as for a batch run.
2. Start or reuse the reconnectable allocation from the local repository:

   ```bash
   bash nscc/interactive_session.sh start [a100|h100]
   ```

   This synchronizes the current local tree, starts `qsub -I` inside a persistent
   login-node tmux session (`spib-a100` or `spib-gpu`), or reuses that session.
   It requests one GPU and `walltime=06:00:00` by default; target-specific CPU
   and memory policy comes from `target_config.sh`. A login-node
   monitor automatically sources `interactive_setup.sh` after the compute-node
   prompt is ready; never inject setup while the job is still queued.
3. Check readiness with `bash nscc/interactive_session.sh status [target]`. A queued
   allocation is normal. Once ready, attach through a persistent PTY:

   ```bash
   bash nscc/interactive_session.sh attach [a100|h100]
   ```

4. After attach, verify the pane contains `SPIB interactive environment ready`
   and the prompt starts with `(spib)` in the runtime checkout. Interactive
   startup deliberately does not import PyTorch or probe the GPU. If GPU
   diagnostics are needed, run `spib_gpu_check` explicitly. If auto-init is
   marked failed, source `nscc/interactive_setup.sh` manually as a recovery
   fallback. Then run the requested configuration:

   ```bash
   spib_run examples/Four_Well_hsic_config.ini
   ```

   `spib_run` gives every execution a unique `interactive-<job-id>-<timestamp>`
   log and result directory under project storage, streams the training output,
   and records `exit_status`.
5. For another local edit during the same allocation: wait for the current
   Python process to finish, edit and check only on the Mac, run
   `bash nscc/sync_code.sh [target]`, then call `spib_run` again in the existing compute
   shell. Never sync over source files while a run is still reading them.
6. Detach from tmux instead of exiting the compute shell when the user plans to
   continue debugging. The allocation survives local SSH/Codex disconnects and
   can be reattached on a later turn. Keep it until walltime expiry; cancel it
   only on explicit request, and report the remaining allocation when known.

Do not confuse seeing a GPU model with having requested GPU resources; require
the PBS allocation. Use `spib_gpu_check` when diagnosing CUDA availability;
normal interactive startup skips this costly import for responsive debugging.

## Saved-version experiment workflow

After the user explicitly asks to save the code and the reviewed commit has
been pushed to `personal/hsic-spib`, submit that exact commit or tag from the
local repository:

```bash
bash nscc/submit_version.sh <commit-or-tag> <config.ini> [a100|h100]
```

Then parse the returned job ID and monitor it with `nscc/watch_job.sh` exactly
as for a working run. Version mode fetches the pushed ref and archives the exact
commit, so synchronized uncommitted files in the runtime mirror are excluded.

## Save and recover versions

When saving is explicitly requested:

1. inspect all relevant diffs and exclude papers, data, results, and unrelated
   user changes;
2. choose the next semantic version and run
   `bash scripts/prepare_version.sh <version> "<summary>"`;
3. review `VERSION`, `CHANGELOG.md`, and the selected source files;
4. selectively stage, commit with the version in the message, and push
   `personal hsic-spib`;
5. create and push an annotated `HSIC-SPIB_V<version>` tag only when the user
   explicitly calls the checkpoint a milestone or requests a tag; honor an
   exact alternative tag name supplied by the user.

For recovery, inspect `git log`, `git show`, and the changelog first. Prefer a
new branch or selective `git restore --source <ref> -- <path>` after explicit
confirmation; never use destructive reset over user-owned changes.

## Existing jobs

- Monitor: `bash nscc/watch_job.sh <job-id> [a100|h100]`.
- List: use `bash nscc/status_summary.sh all` or query the selected host.
- Inspect: use `qstat -f <job-id>` and the persistent project log/results.
- Cancel only on explicit request with `qdel <job-id>` after validating the ID.

## Final response

Lead with the outcome. Include:

- what changed and which local checks passed;
- execution mode, config, job ID, and terminal status;
- log and result paths;
- important metrics, outputs, or the exact failure cause;
- whether an automatic fix/rerun occurred;
- confirmation that Git was left uncommitted/unpushed unless saving was asked.

Keep commands out of the final response unless the user needs to take a manual
action. The normal experience should be: request the change once, then receive
progress updates and a completed NSCC result.

End every final response with compact lines generated from a fresh
`bash nscc/status_summary.sh all` check when practical. Show both targets when
reachable and clearly mark an unavailable target:

- `ASPIRE2A-A100：...`
- `ASPIRE2A+-H100：...`

Keep this as the final line, even for local-only work or conceptual answers.
Do not infer an active allocation from an old conversation message.
