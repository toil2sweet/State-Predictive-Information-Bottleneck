"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import numpy as np
import torch
import os
import sys
import configparser
import json
import random

import SPIB
import SPIB_training
import plot_state_labels
import plot_transition_states
import plot_spib_plus

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")


def test_model_advanced():
    # Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    # Keep the historical default for local runs, but allow NSCC batch jobs to
    # place large results outside the $HOME quota.
    base_path = os.environ.get("SPIB_OUTPUT_DIR", "SPIB")
    

    # If there is a configuration file, import the configuration file
    # Otherwise, an error will be reported
    if '-config' in sys.argv:
        config = configparser.ConfigParser(allow_no_value=True)

        config.read(sys.argv[sys.argv.index('-config') + 1])

        # Model parameters
        # Time delay delta t in terms of # of minimal time resolution of the trajectory data
        dt_list = json.loads(config.get("Model Parameters","dt"))
        
        # By default, we use all the all the data to train and test our model
        t0 = 0 
        
        # Dimension of RC or bottleneck
        RC_dim_list = json.loads(config.get("Model Parameters","d"))
        
        # Encoder type ('Linear' or 'Nonlinear')
        if config.get("Model Parameters","encoder_type")=='Nonlinear':
            encoder_type = 'Nonlinear'
        else:
            encoder_type = 'Linear'

        # Number of nodes in each hidden layer of the encoder
        neuron_num1_list = json.loads(config.get("Model Parameters","neuron_num1"))
        # Number of nodes in each hidden layer of the encoder
        neuron_num2_list = json.loads(config.get("Model Parameters","neuron_num2"))
        
        
        # Training parameters
        batch_size = int(config.get("Training Parameters","batch_size"))

        # Threshold in terms of the change of the predicted state population for measuring the convergence of training
        threshold = float(config.get("Training Parameters","threshold"))

        # Number of epochs with the change of the state population smaller than the threshold after which this iteration of training finishes
        patience = int(config.get("Training Parameters","patience"))

        # Minimum refinements
        refinements = int(config.get("Training Parameters","refinements"))

        # Optional SGD subset per epoch. 0 / omitted = use all training frames.
        # Label refinement, state-number, and TS detection still use the full data.
        epoch_sample_size = 0
        if config.has_option("Training Parameters", "epoch_sample_size"):
            epoch_sample_raw = config.get("Training Parameters", "epoch_sample_size")
            if epoch_sample_raw is not None and str(epoch_sample_raw).strip():
                epoch_sample_size = int(epoch_sample_raw)
            
        # By default, we save the model every 10000 steps
        log_interval = int(config.get("Training Parameters","log_interval"))
        
        # Period of learning rate decay
        lr_scheduler_step_size = int(config.get("Training Parameters","lr_scheduler_step_size"))

        # Multiplicative factor of learning rate decay. Default: 1 (No learning rate decay)
        lr_scheduler_gamma = float(config.get("Training Parameters","lr_scheduler_gamma"))

        # learning rate of Adam optimizer
        learning_rate_list = json.loads(config.get("Training Parameters","learning_rate"))
        
        # Hyper-parameter beta
        beta_list = json.loads(config.get("Training Parameters","beta"))

        # Optional HSIC-SPIB section (defaults preserve original SPIB)
        hsic_config = SPIB_training.default_hsic_config()
        if config.has_section("HSIC-SPIB"):
            if config.has_option("HSIC-SPIB", "loss_mode"):
                hsic_config["loss_mode"] = config.get("HSIC-SPIB", "loss_mode")
            if config.has_option("HSIC-SPIB", "lambda_y"):
                hsic_config["lambda_y"] = float(config.get("HSIC-SPIB", "lambda_y"))
            if config.has_option("HSIC-SPIB", "beta_x"):
                hsic_config["beta_x"] = float(config.get("HSIC-SPIB", "beta_x"))
            if config.has_option("HSIC-SPIB", "normalized_hsic"):
                hsic_config["normalized_hsic"] = config.get("HSIC-SPIB", "normalized_hsic") == "True"
            if config.has_option("HSIC-SPIB", "decoder_on_mean"):
                hsic_config["decoder_on_mean"] = config.get("HSIC-SPIB", "decoder_on_mean") == "True"
            if config.has_option("HSIC-SPIB", "kernel_z"):
                hsic_config["kernel_z"] = config.get("HSIC-SPIB", "kernel_z")
            if config.has_option("HSIC-SPIB", "kernel_x"):
                hsic_config["kernel_x"] = config.get("HSIC-SPIB", "kernel_x")
            if config.has_option("HSIC-SPIB", "kernel_y"):
                hsic_config["kernel_y"] = config.get("HSIC-SPIB", "kernel_y")
            if config.has_option("HSIC-SPIB", "DetectTransitionStates"):
                hsic_config["DetectTransitionStates"] = (
                    config.get("HSIC-SPIB", "DetectTransitionStates") == "True")
            if config.has_option("HSIC-SPIB", "eps_ts"):
                hsic_config["eps_ts"] = float(config.get("HSIC-SPIB", "eps_ts"))
            if config.has_option("HSIC-SPIB", "ts_require_cross_state"):
                hsic_config["ts_require_cross_state"] = (
                    config.get("HSIC-SPIB", "ts_require_cross_state") == "True")
            if config.has_option("HSIC-SPIB", "ts_window"):
                hsic_config["ts_window"] = int(config.get("HSIC-SPIB", "ts_window"))
            if config.has_option("HSIC-SPIB", "fe_beta"):
                hsic_config["fe_beta"] = float(config.get("HSIC-SPIB", "fe_beta"))
            if config.has_option("HSIC-SPIB", "fe_vmax"):
                hsic_config["fe_vmax"] = float(config.get("HSIC-SPIB", "fe_vmax"))
            if config.has_option("HSIC-SPIB", "ts_potential"):
                hsic_config["ts_potential"] = config.get("HSIC-SPIB", "ts_potential")
            if config.has_option("HSIC-SPIB", "eps_rho"):
                hsic_config["eps_rho"] = float(config.get("HSIC-SPIB", "eps_rho"))
            if config.has_option("HSIC-SPIB", "encoder_var_mode"):
                hsic_config["encoder_var_mode"] = config.get("HSIC-SPIB", "encoder_var_mode")
        
        # Import data

        # Path to the trajectory data
        traj_data_path = config.get("Data","traj_data")
        traj_data_path = traj_data_path.replace('[','').replace(']','')
        traj_data_path = [os.path.expandvars(path.strip()) for path in traj_data_path.split(',')]

        # Load the data
        traj_data_list = [torch.from_numpy(np.load(file_path)).float().to(device) for file_path in traj_data_path]

        # Optional fixed DataNormalize (mean/std npy paths for HSIC-SPIB+ / protein)
        data_transform = None
        if config.has_option("Data", "data_mean") and config.has_option("Data", "data_std"):
            mean_path = config.get("Data", "data_mean")
            std_path = config.get("Data", "data_std")
            if mean_path is not None and std_path is not None:
                mean_path = os.path.expandvars(str(mean_path).strip())
                std_path = os.path.expandvars(str(std_path).strip())
                if mean_path and std_path:
                    data_transform = SPIB.DataNormalize(np.load(mean_path), np.load(std_path))
        
        # Path to the initial state labels
        initial_labels_path = config.get("Data","initial_labels")
        initial_labels_path = initial_labels_path.replace('[','').replace(']','')
        initial_labels_path = [os.path.expandvars(path.strip()) for path in initial_labels_path.split(',')]
        
        traj_labels_list = [torch.from_numpy(np.load(file_path)).float().to(device) for file_path in initial_labels_path]
        
        output_dim = traj_labels_list[0].shape[1]
        
        assert len(traj_data_list)==len(traj_labels_list)

        # Path to the weights of the samples
        traj_weights_path = config.get("Data", "traj_weights")
        if traj_weights_path is None or str(traj_weights_path).strip() == "":
            traj_weights_list = None
            IB_path = os.path.join(base_path, "Unweighted")
        else:
            traj_weights_path = traj_weights_path.replace('[','').replace(']','')
            traj_weights_path = [os.path.expandvars(path.strip()) for path in traj_weights_path.split(',')]
        
            traj_weights_list = [torch.from_numpy(np.load(file_path)).float().to(device) for file_path in traj_weights_path]
            IB_path = os.path.join(base_path, "Weighted")
            assert len(traj_weights_list)==len(traj_labels_list)

        
        # Other controls

        # Random seed
        seed_list = json.loads(config.get("Other Controls","seed"))

        # Whether to refine the labels during the training process
        if config.get("Other Controls","UpdateLabel") == 'True':
            UpdateLabel = True
        else:
            UpdateLabel = False
        
        # Whether save trajectory results
        if config.get("Other Controls","SaveTrajResults") == 'True':
            SaveTrajResults = True
        else:
            SaveTrajResults = False

        # Whether to plot/save learned state labels (SPIB_Demo style) under fig/
        # Default False: plain SPIB configs (e.g. Double_Well_CTC) do not auto-plot.
        # HSIC configs should set SaveLabelPlot = True explicitly.
        if config.has_option("Other Controls", "SaveLabelPlot"):
            SaveLabelPlot = config.get("Other Controls", "SaveLabelPlot") == "True"
        else:
            SaveLabelPlot = False

        if config.has_option("Other Controls", "fig_dir"):
            fig_dir = config.get("Other Controls", "fig_dir")
        else:
            fig_dir = "fig"
        # The environment override is useful for batch jobs and takes
        # precedence over a machine-specific path in a config file.
        fig_dir = os.environ.get("SPIB_FIG_DIR", fig_dir)

    else:
        print("Pleast input the config file!")
        return

    
    
    # Train and Test our model
    # ------------------------------------------------------------------------------

    final_result_path = IB_path + '_result.dat'
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
    print("Final Result", file=open(final_result_path, 'w'))
    
    for seed in seed_list:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        for dt in dt_list:
            data_init_list = [] 
            if traj_weights_list == None:
                for i in range(len(traj_data_list)):
                    data_init_list+=[SPIB_training.data_init(t0, dt, traj_data_list[i], traj_labels_list[i], None)]
                train_data_weights = None
                test_data_weights = None
            else:
                for i in range(len(traj_data_list)):
                    data_init_list+=[SPIB_training.data_init(t0, dt, traj_data_list[i], traj_labels_list[i], traj_weights_list[i])]

                train_data_weights = torch.cat([data_init_list[i][4] for i in range(len(traj_data_list))], dim=0)
                test_data_weights = torch.cat([data_init_list[i][8] for i in range(len(traj_data_list))], dim=0)

            data_shape = data_init_list[0][0]
            train_past_data = torch.cat([data_init_list[i][1] for i in range(len(traj_data_list))], dim=0)
            train_future_data = torch.cat([data_init_list[i][2] for i in range(len(traj_data_list))], dim=0)
            train_data_labels = torch.cat([data_init_list[i][3] for i in range(len(traj_data_list))], dim=0)

            test_past_data = torch.cat([data_init_list[i][5] for i in range(len(traj_data_list))], dim=0)
            test_future_data = torch.cat([data_init_list[i][6] for i in range(len(traj_data_list))], dim=0)
            test_data_labels = torch.cat([data_init_list[i][7] for i in range(len(traj_data_list))], dim=0)

            for RC_dim in RC_dim_list:
                for neuron_num1 in neuron_num1_list:
                    for neuron_num2 in neuron_num2_list:
                        for beta in beta_list:
                            for learning_rate in learning_rate_list:

                                output_path = IB_path + "_d=%d_t=%d_b=%.4f_learn=%f_%s" \
                                    % (RC_dim, dt, beta, learning_rate, hsic_config["loss_mode"])

                                run_hsic_config = dict(hsic_config)
                                run_hsic_config["beta_kl"] = beta

                                IB = SPIB.SPIB(
                                    encoder_type, RC_dim, output_dim, data_shape, device,
                                    UpdateLabel, neuron_num1, neuron_num2,
                                    data_transform=data_transform,
                                    encoder_var_mode=hsic_config.get(
                                        "encoder_var_mode", "input_dependent"))
                                
                                IB.to(device)
                                
                                # use the training set to initialize the pseudo-inputs
                                IB.init_representative_inputs(train_past_data, train_data_labels)

                                train_result = False
                                
                                train_result = SPIB_training.train(IB, beta, train_past_data, train_future_data, \
                                                                train_data_labels, train_data_weights, test_past_data, test_future_data, \
                                                                    test_data_labels, test_data_weights, learning_rate, lr_scheduler_step_size, lr_scheduler_gamma,\
                                                                        batch_size, threshold, patience, refinements, output_path, \
                                                                            log_interval, device, seed, hsic_config=run_hsic_config,
                                                                            epoch_sample_size=epoch_sample_size)
                                
                                if train_result:
                                    return
                                
                                SPIB_training.output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_weights, \
                                                                test_past_data, test_future_data, test_data_labels, test_data_weights, batch_size, \
                                                                    output_path, final_result_path, dt, beta, learning_rate, seed,
                                                                    hsic_config=run_hsic_config)

                                for i in range(len(traj_data_list)):
                                    IB.save_traj_results(traj_data_list[i], batch_size, output_path, SaveTrajResults, i, seed)

                                    # After state number: decoder-K_i transition-state identification
                                    if SaveTrajResults and run_hsic_config.get("DetectTransitionStates", False):
                                        pred_path = output_path + "_traj%d_data_prediction%d.npy" % (i, seed)
                                        pop_path = output_path + "_traj%d_state_population%d.npy" % (i, seed)
                                        if os.path.isfile(pred_path):
                                            SPIB_training.save_and_report_transition_states(
                                                pred_path,
                                                population_path=pop_path if os.path.isfile(pop_path) else None,
                                                output_prefix=output_path + "_traj%d_ts%d" % (i, seed),
                                                eps_ts=float(run_hsic_config.get("eps_ts", 0.1)),
                                                require_cross_state=bool(
                                                    run_hsic_config.get("ts_require_cross_state", True)),
                                                window=int(run_hsic_config.get("ts_window", 1)),
                                                log_path=final_result_path)

                                    # Plot learned state labels (same style as SPIB_Demo.ipynb)
                                    if SaveTrajResults and SaveLabelPlot:
                                        labels_path = output_path + "_traj%d_labels%d.npy" % (i, seed)
                                        if os.path.isfile(labels_path):
                                            traj_np = traj_data_list[i].detach().cpu().numpy()
                                            labels_np = np.load(labels_path)
                                            fig_prefix = "HSIC_SPIB_plus" if (
                                                run_hsic_config.get("eps_rho", 0) not in (0, 0.0)
                                                or run_hsic_config.get("encoder_var_mode") == "isotropic"
                                                or data_transform is not None
                                                or "plus" in os.path.basename(
                                                    sys.argv[sys.argv.index('-config') + 1]).lower()
                                            ) else "HSIC_SPIB"
                                            fig_name = "%s_learned_labels_d=%d_t=%d_b=%.4f_traj%d_seed%d.png" % (
                                                fig_prefix, RC_dim, dt, beta, i, seed)
                                            if hsic_config.get("loss_mode"):
                                                fig_name = "%s_%s" % (hsic_config["loss_mode"], fig_name)
                                            fig_path = os.path.join(fig_dir, fig_name)
                                            title = "SPIB learned labels (dt=%d, traj=%d)" % (dt, i)
                                            pot = run_hsic_config.get("ts_potential", None)
                                            if pot in ("double_well", "dw"):
                                                title = "HSIC-SPIB double-well labels (dt=%d)" % dt
                                            elif pot in ("four_well", "fw"):
                                                title = "HSIC-SPIB four-well labels (dt=%d)" % dt
                                            elif pot in ("muller", "muller_brown", "mb"):
                                                title = "HSIC-SPIB+ Müller labels (dt=%d)" % dt
                                            elif pot in ("trpcage", "trp_cage", "protein"):
                                                title = "HSIC-SPIB+ Trp-cage labels (dt=%d)" % dt
                                            # For high-dim protein traj, plot in SPIB latent if available
                                            plot_traj = traj_np
                                            mean_rep_path = output_path + "_traj%d_mean_representation%d.npy" % (i, seed)
                                            if traj_np.ndim == 2 and traj_np.shape[1] > 2 and os.path.isfile(mean_rep_path):
                                                plot_traj = np.load(mean_rep_path)
                                                if pot in ("trpcage", "trp_cage", "protein"):
                                                    title = "HSIC-SPIB+ Trp-cage latent labels (dt=%d)" % dt
                                            saved, active, pop = plot_state_labels.plot_learned_state_labels(
                                                plot_traj, labels_np, fig_path, title=title)
                                            print("Saved learned state-label plot: %s (n_states=%d, indices=%s)" % (
                                                saved, len(active), active.tolist()))
                                            print("Saved learned state-label plot: %s (n_states=%d, indices=%s)" % (
                                                saved, len(active), active.tolist()),
                                                file=open(final_result_path, 'a'))

                                            # HSIC-SPIB+: latent FE / labels + state-number vs refinement
                                            if fig_prefix == "HSIC_SPIB_plus" and os.path.isfile(mean_rep_path):
                                                z_lat = np.load(mean_rep_path)
                                                hist_path = output_path + "_convergence_history%d.npy" % seed
                                                conv = np.load(hist_path) if os.path.isfile(hist_path) else (
                                                    IB.convergence_history if len(IB.convergence_history) > 0 else None)
                                                plus_prefix = "%s_%s_d=%d_t=%d_b=%.4f_traj%d_seed%d" % (
                                                    hsic_config.get("loss_mode", "hsic"),
                                                    fig_prefix, RC_dim, dt, beta, i, seed)
                                                plus_figs = plot_spib_plus.plot_plus_summary(
                                                    z_lat if z_lat.shape[1] >= 2 else None,
                                                    labels_np, conv, fig_dir, plus_prefix,
                                                    fe_beta=float(run_hsic_config.get("fe_beta", 1.0)),
                                                    fe_vmax=run_hsic_config.get("fe_vmax", None),
                                                    dpi=150)
                                                for kind, path in plus_figs:
                                                    msg = "Saved HSIC-SPIB+ %s plot: %s" % (kind, path)
                                                    print(msg)
                                                    print(msg, file=open(final_result_path, 'a'))

                                            # Three TS-annotated figures (labels / free energy / analytical potential)
                                            # Skip analytical-potential TS for protein (high-dim / no closed form)
                                            ts_mask_path = output_path + "_traj%d_ts%d_mask.npy" % (i, seed)
                                            if (run_hsic_config.get("DetectTransitionStates", False)
                                                    and os.path.isfile(ts_mask_path)
                                                    and pot not in ("trpcage", "trp_cage", "protein")):
                                                ts_mask = np.load(ts_mask_path)
                                                name_prefix = "HSIC_SPIB_TS_d=%d_t=%d_b=%.4f_traj%d_seed%d" % (
                                                    RC_dim, dt, beta, i, seed)
                                                if hsic_config.get("loss_mode"):
                                                    name_prefix = "%s_%s" % (
                                                        hsic_config["loss_mode"], name_prefix)
                                                pot = run_hsic_config.get("ts_potential", None)
                                                if pot is None or str(pot).strip() == "":
                                                    print(
                                                        "Skip potential+TS plot: set [HSIC-SPIB] "
                                                        "ts_potential = four_well|double_well|muller")
                                                    pot = None
                                                ts_figs = plot_transition_states.plot_all_ts_figures(
                                                    traj_np, labels_np, ts_mask, fig_dir, name_prefix,
                                                    fe_beta=float(run_hsic_config.get("fe_beta", 3.0)),
                                                    potential=pot,
                                                    fe_vmax=run_hsic_config.get("fe_vmax", None),
                                                    dpi=150)
                                                for kind, path, n_ts in ts_figs:
                                                    msg = "Saved %s plot: %s (n_TS=%d)" % (
                                                        kind, path, n_ts)
                                                    print(msg)
                                                    print(msg, file=open(final_result_path, 'a'))
                                
                                IB.save_representative_parameters(output_path, seed)


if __name__ == '__main__':
    
    test_model_advanced()
    

    
    
