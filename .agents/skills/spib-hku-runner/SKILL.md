---
name: spib-hku-runner
description: Local HSIC-SPIB/CTC modification and reconnectable HKU CS GPU-farm execution via gpu-interactive. Use only when the user asks to run on the HKU server. Do not use for NSCC PBS or DNA Slurm.
---

# SPIB HKU Runner

Treat one natural-language request as one end-to-end operation. Do not change
or reuse the `nscc/` PBS helpers for this target.

## Fixed context

- Use this repository as the local editable source on branch `hsic-spib`.
- Trigger only when the user says `hku服务器上执行`, `HKU`, `港大`, or `gpu3`.
- Unspecified execution still defaults to NSCC A100.
- SSH host alias is `HKUCDS_GPU_Farm` (`lidepeng@gpu3gate1.cs.hku.hk`).
- `/userhome/cs3/lidepeng/TS` is a multi-project workspace. Never rsync onto
  that root. The single runtime checkout is
  `/userhome/cs3/lidepeng/TS/State-Predictive-Information-Bottleneck`.
- Leave `State-Predictive-Information-Bottleneck_v1/v2/v3`, `spib`,
  `ts-dar`, and `CTClustering` untouched.
- After `gpu-interactive`, use the default Conda `base` environment at
  `/userhome/cs3/lidepeng/anaconda3`. Do not use `macil` or `CL`, and do
  not copy an NSCC environment.
- There is no separate HKU project-storage root. Trajectories, logs, and
  results stay under the runtime checkout. Do not expand
  `${NSCC_PROJECT_ROOT}` on this target.
- Enter a GPU node with `gpu-interactive` (Slurm `srun --gpus=1 --pty bash`).
  Never train on `gpu3gate1`.
- Default config is `examples/Four_Well_hsic_hku_config.ini`.
- After every HKU execution, download any produced figures with
  `hku/fetch_figures.sh` into `fig/<system>-<job>-<version>-<MMDDTHHMM>/`.
  Do not leave figures only on the server.
- Edit source only on the local Mac.

## Interpret the request

- “hku服务器上执行” selects this interactive workflow.
- “只同步” means `bash hku/sync_code.sh` and stop after verification.
- There is no HKU batch/PBS snapshot workflow.

## Modify-and-run workflow

1. Inspect local diffs and run `python -m py_compile` / `bash -n` as for NSCC.
2. Verify SSH:

   ```bash
   ssh -o BatchMode=yes -o ConnectTimeout=15 HKUCDS_GPU_Farm true
   ```

3. Execute from the local Mac so figures are fetched automatically:

   ```bash
   bash hku/run.sh examples/Four_Well_hsic_hku_config.ini
   ```

   This synchronizes the working tree plus Four-Well and Double-Well
   `traj_gen` files, double-well `double-well_CTC/*.npy` trajectories/labels, starts
   or reuses `gpu-interactive` inside login-node tmux session `spib-hku`,
   sources `hku/interactive_setup.sh` when the compute prompt is ready, runs
   the config, and copies any PNG/PDF/SVG files into local
   `fig/<system>-<job>-<version>-<MMDDTHHMM>/`.

4. Attach only when terminal inspection is needed:

   ```bash
   bash hku/interactive_session.sh attach
   ```

   If a run was started with `spib_run` inside that shell instead of
   `hku/run.sh`, fetch figures afterwards:

   ```bash
   bash hku/fetch_figures.sh
   ```

5. After another local edit: wait for the current Python process to finish,
   then run `bash hku/run.sh <config>` again.

Keep the allocation until the user asks to cancel it (`scancel` / exit the
gpu-interactive shell). Detach tmux instead of exiting when debugging continues.

## Final response

Lead with the outcome. Include mode, config, Slurm job if present, log,
result paths, and whether figures were downloaded into
`fig/<system>-<job>-<version>-<MMDDTHHMM>/`. End
with a fresh `bash hku/status_summary.sh` line and the existing NSCC
status lines.
