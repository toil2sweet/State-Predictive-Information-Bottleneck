# Changelog

This file records accepted HSIC-SPIB code checkpoints. Git commits are the
authoritative recovery points; annotated tags are reserved for research
milestones.

## Unreleased

## 0.5.0 - 2026-09-01

- Add shared encoder and multi-head decoders for Trp-cage.

- Identify Müller transition states with population-normalized decoder margin K_i/ρ_i so the isocommittor is not pinned to the shallow well.
- Rank Müller decoder-margin TS overlays by high empirical F or analytical V per state pair without changing saved ts_mask.
- Generate unused Müller xy-kmeans K=30 labels alongside the loaded K=20 set.
- Add Müller xy-kmeans K=20 labels and point both configs plus HKU sync at them.
- Align Müller HSIC configs with Four-Well/Double-Well Linear 16/16, eps_ts=0.005, and CTC-style TS.
- Keep only Müller xy-kmeans K=10 labels and point both Müller configs at them.
- Regenerate Müller 10-state x-bin and xy-kmeans labels for HSIC-SPIB and HSIC-SPIB+.
- Draw Trp-cage MTL free-energy+TS with the tutorial3 latent recipe on one shared (IB_0, IB_1) grid.
- Add Trp-cage HKU MTL config for dt=[50, 100, 500] and plot combined lag figures in the shared 2D SPIB latent.
- Add shared-encoder multi-lag HSIC-SPIB training with per-Δt decoders and combined Four-Well figures.
- Align Trp-cage HSIC-SPIB+ with the converged coarse SPIB protocol and reduce normalized-HSIC regularization for stable refinement.
- Align Trp-cage SPIB refinement with tutorial3 loss tolerance and contiguous-block validation split.
- Enable Trp-cage decoder-margin transition-state detection with SPIB-latent free-energy and metastable-state overlays.
- Sync prepared Trp-cage arrays to HKU and keep spib_run at the runtime root.
- Add tutorial3-style Trp-cage FE/label plots and NSCC project-storage data sync for interactive spib_run.
- Add original 2024 SPIB Trp-cage baseline from local DESRES Anton DCDs.
- Milestone tag: `shared_encoder_multihead_decoders_trpcage`.

## 0.4.0 - 2026-08-26

- Add CTC-style three-level transition-state representatives, comparison plots, and HKU figure directories.

- Omit the CTC-style title prefix from four-well free-energy+TS comparison plots.
- Store fetched HKU figures in fig/<system>-<job>-<version>-<MMDDTHHMM>.
- Add config-driven CTC-style transition-event representatives and density-ranked top-k comparison plots.
- Milestone tag: `3_level_TS`.

## 0.3.0 - 2026-08-24

- Add reproducible Double-Well and Four-Well trajectory generation with improved SPIB/CTC-style plots and GPU workflows.

- Point the HKU double-well config at the generated Langevin trajectory and five-state labels under traj_gen.
- Draw double-well potential_with_TS with the CTC analytical-landscape background from plot_energy_landscape_likeCTC.py.
- Set the CTC double-well energy-landscape window from the generated Langevin trajectory, matching the four-well plot.
- Draw the CTC double-well analytical potential from the transition_state.ipynb energy-landscape background without transition-state markers.
- Sync double-well CTC trajectory and label npy files to the HKU runtime checkout.
- Add CTC double-well Langevin trajectory generation, five-state labels, and SPIB/CTC-style landscape plots.
- Keep the Four-Well trajectory generator and its generated arrays together under traj_gen.
- Organize Four-Well utilities and NSCC documentation into dedicated directories with updated path loading.
- Download HKU run figures into the local fig directory after each execution.
- Draw Four-Well free-energy+TS and potential+TS backgrounds with the standalone SPIB and CTC plot recipes.
- Use HKU conda base after gpu-interactive and keep data, logs, and results in the TS checkout.
- Add a parallel HKU gpu-interactive runner that syncs the local tree into a dedicated TS checkout without changing NSCC.
- Run the revised Four-Well plots and HSIC-SPIB workflow on the 60,000-time-unit seed2026 trajectory.
- Add one-command A100/H100 target selection with A100 default, isolated environments, data, logs, results, and figures
- Make four-well trajectory generation take a CLI seed (default 0) that names the output files, and show tqdm progress.
- Draw four-well HSIC-SPIB potential_with_TS with the CTC landscape recipe and free_energy_with_TS with the SPIB Fig. 5(b) recipe.
- Add SPIB Fig. 5(a)/(b)-style four-well potential and free-energy plots from the generated trajectory.
- Set four-well CTC-style colorbar ticks to 0,1,2,3 and match colorbar height to the y-axis.
- Honor explicit four-well CTC-style xlim/ylim without auto-squaring so the analytical potential can serve as an HSIC-SPIB TS background.
- Draw a square CTC-style four-well analytical potential (no TS) with explicit xlim/ylim controls.
- Reframe the four-well CTC-style potential to a near-square equal-aspect window that fills with complete basins.
- Add a CTC-style four-well analytical-potential plot (no TS markers) from the generated trajectory as an HSIC-SPIB background.
- Match the four-well CTC-style potential plot to Fig. 2(F): landscape jet panel, 0-3 energy, window that fills with all four wells.
- Tighten four-well CTC-style potential plot to a square equal-aspect window so the four wells fill the panel.
- Expand the four-well potential window so the 0–3 color scale shows complete basins.
- Use a square equal-aspect window and well-to-barrier color scale for the four-well CTC-style potential plot.
- Add a CTC Fig. 2(F)-style four-well analytical-potential plot with no TS overlay for later HSIC-SPIB backgrounds.
- Speed up long-trajectory HSIC-SPIB training by sharing Z kernels, skipping unused KL, and optional per-epoch SGD subsampling while keeping full-data labels and TS.
- Automatically sync completed NSCC figures into local job-and-version directories.
- Run HSIC-SPIB Four-Well configuration on the 60,000-time-unit trajectory and labels in NSCC project storage.
- Match the default four-well production trajectory length to the SPIB paper's 60,000 reduced time units.
- Automatically retrieve completed NSCC batch figures into the local fig directory
- Ignore terminal PBS jobs in the compact NSCC status summary
- Skip automatic interactive PyTorch, CUDA, and GPU validation; retain it as optional spib_gpu_check
- Remove obsolete secondary-checkout compatibility scripts, aliases, variables, and documentation references.
- Automatically activate the SPIB Conda environment when a reconnectable NSCC interactive GPU shell becomes ready.
- Generate working-tree status and recovery patches against the true local base commit rather than the runtime mirror Git HEAD.
- Retire the legacy secondary NSCC checkout after successful batch and interactive validation, retaining a checksummed recovery archive.
- Add a minimal NSCC GPU probe that verifies PBS, the SPIB environment, and CUDA without training.
- Keep monitoring PBS E-to-R launch retries and report scheduler exit status.
- Submit the PBS harness from the short runtime path while preserving immutable source snapshots.
- Add lightweight Unreleased change logging for high-frequency development.
- Consolidate NSCC execution around one disposable runtime mirror while
  retaining immutable per-job source snapshots.
- Add explicit version metadata to interactive and batch results.

## 0.2.0 - HSIC-SPIB/CTC adaptation milestone

- Add HSIC-SPIB configurations and CTC adaptations.
- Historical tag: `v0.2.0-hsic-spib-ctc`.

## 0.1.0 - HSIC-SPIB baseline

- Establish the baseline HSIC-SPIB implementation.
- Historical tag: `v0.1.0-hsic-spib`.
