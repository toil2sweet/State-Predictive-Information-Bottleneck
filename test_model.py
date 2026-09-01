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
import random

import SPIB
import SPIB_training
import plot_state_labels
import plot_transition_states

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
default_device = torch.device("cpu")


def test_model():
    # Settings
    # ------------------------------------------------------------------------------
    # By default, we save all the results in subdirectories of the following path.
    # Keep the historical default for local runs, but allow NSCC batch jobs to
    # place large results outside the $HOME quota.
    base_path = os.environ.get("SPIB_OUTPUT_DIR", "SPIB")
    
    # Model parameters
    # Time delay delta t in terms of # of minimal time resolution of the trajectory data
    if '-dt' in sys.argv:
        dt = int(sys.argv[sys.argv.index('-dt') + 1])
    else:
        dt = 10
    
    # By default, we use all the all the data to train and test our model
    t0 = 0 
    
    # Dimension of RC or bottleneck
    if '-d' in sys.argv:
        RC_dim = int(sys.argv[sys.argv.index('-d') + 1])
    else:
        RC_dim = 2
    
    # Encoder type ('Linear' or 'Nonlinear')
    if '-encoder_type' in sys.argv and (sys.argv[sys.argv.index('-encoder_type') + 1])=='Nonlinear':
        encoder_type = 'Nonlinear'
    else:
        encoder_type = 'Linear'

    # Number of nodes in each hidden layer of the encoder
    if '-n1' in sys.argv:
        neuron_num1 = int(sys.argv[sys.argv.index('-n1') + 1])
    else:
        neuron_num1 = 16
    # Number of nodes in each hidden layer of the encoder
    if '-n2' in sys.argv:
        neuron_num2 = int(sys.argv[sys.argv.index('-n2') + 1])
    else:
        neuron_num2 = 16
    
    
    # Training parameters
    
    if '-bs' in sys.argv:
        batch_size = int(sys.argv[sys.argv.index('-bs') + 1])
    else:
        batch_size = 2048

    # Threshold in terms of the change of the predicted state population for measuring the convergence of the training
    if '-threshold' in sys.argv:
        threshold = float(sys.argv[sys.argv.index('-threshold') + 1])
    else:
        threshold = 0.01

    # Number of epochs with the change of the state population smaller than the threshold after which this iteration of the training finishes
    if '-patience' in sys.argv:
        patience = int(sys.argv[sys.argv.index('-patience') + 1])
    else:
        patience = 0

    # Minimum refinements
    if '-refinements' in sys.argv:
        refinements = int(sys.argv[sys.argv.index('-refinements') + 1])
    else:
        refinements = 0
        
    # By default, we save the model every 10000 steps
    log_interval = 10000 
    
    # By default, there is no learning rate decay
    lr_scheduler_step_size = 1
    lr_scheduler_gamma = 1

    # Initial learning rate of Adam optimizer
    if '-lr' in sys.argv:
        learning_rate = float(sys.argv[sys.argv.index('-lr') + 1])
    else:
        learning_rate = 1e-3
    
    # Hyper-parameter beta
    if '-b' in sys.argv:
        beta = float(sys.argv[sys.argv.index('-b') + 1])
    else:
        beta = 1e-3

    # HSIC-SPIB options (state-number + decoder-K_i transition-state detection)
    if '-loss_mode' in sys.argv:
        loss_mode = sys.argv[sys.argv.index('-loss_mode') + 1]
    else:
        loss_mode = 'original_spib'

    if '-lambda_y' in sys.argv:
        lambda_y = float(sys.argv[sys.argv.index('-lambda_y') + 1])
    else:
        lambda_y = 0.0

    if '-beta_x' in sys.argv:
        beta_x = float(sys.argv[sys.argv.index('-beta_x') + 1])
    else:
        beta_x = 1.0

    if '-normalized_hsic' in sys.argv:
        normalized_hsic = bool(int(sys.argv[sys.argv.index('-normalized_hsic') + 1]))
    else:
        normalized_hsic = True

    if '-decoder_on_mean' in sys.argv:
        decoder_on_mean = bool(int(sys.argv[sys.argv.index('-decoder_on_mean') + 1]))
    else:
        decoder_on_mean = True

    if '-DetectTransitionStates' in sys.argv:
        DetectTransitionStates = bool(int(sys.argv[sys.argv.index('-DetectTransitionStates') + 1]))
    else:
        DetectTransitionStates = True

    if '-eps_ts' in sys.argv:
        eps_ts = float(sys.argv[sys.argv.index('-eps_ts') + 1])
    else:
        eps_ts = 0.1

    hsic_config = {
        'loss_mode': loss_mode,
        'lambda_y': lambda_y,
        'beta_x': beta_x,
        'beta_kl': beta,
        'normalized_hsic': normalized_hsic,
        'decoder_on_mean': decoder_on_mean,
        'kernel_z': 'rbf',
        'kernel_x': 'rbf',
        'kernel_y': 'delta',
        'DetectTransitionStates': DetectTransitionStates,
        'eps_ts': eps_ts,
        'ts_require_cross_state': True,
        'ts_window': 1,
    }
    
    # Import data
    
    # Path to the initial state labels
    if '-label' in sys.argv:
        initial_label = np.load(sys.argv[sys.argv.index('-label') + 1])
    else:
        print("Pleast input the initial state labels!")
        return
    
    traj_labels = torch.from_numpy(initial_label).float().to(device)
    output_dim = initial_label.shape[1]
    
    # Path to the trajectory data
    if '-traj' in sys.argv:
        traj_data = np.load(sys.argv[sys.argv.index('-traj') + 1])
    else:
        print("Pleast input the trajectory data!")
        return
    
    traj_data = torch.from_numpy(traj_data).float().to(device)
    
    
    # Path to the weights of the samples
    if '-w' in sys.argv:
        traj_weights = np.load(sys.argv[sys.argv.index('-bias') + 1])
        traj_weights = torch.from_numpy(traj_weights).float().to(device)
        IB_path = os.path.join(base_path, "Weighted")
    else:
        traj_weights = None
        IB_path = os.path.join(base_path, "Unweighted")
    
    # Random seed
    if '-seed' in sys.argv:
        seed = int(sys.argv[sys.argv.index('-seed') + 1])
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)    
    else:
        seed = 0
    
    
    # Other controls
    
    # Whether to refine the labels during the training process
    if '-UpdateLabel' in sys.argv:
        UpdateLabel = bool(sys.argv[sys.argv.index('-UpdateLabel') + 1])  
    else:
        UpdateLabel = True
    
    
    # Whether save trajectory results
    if '-SaveTrajResults' in sys.argv:
        SaveTrajResults = bool(sys.argv[sys.argv.index('-SaveTrajResults') + 1])  
    else:
        SaveTrajResults = True

    if '-SaveLabelPlot' in sys.argv:
        SaveLabelPlot = bool(int(sys.argv[sys.argv.index('-SaveLabelPlot') + 1]))
    else:
        SaveLabelPlot = True

    if '-fig_dir' in sys.argv:
        fig_dir = sys.argv[sys.argv.index('-fig_dir') + 1]
    else:
        fig_dir = os.environ.get('SPIB_FIG_DIR', 'fig')
    
    # Train and Test our model
    # ------------------------------------------------------------------------------
    
    final_result_path = IB_path + '_result.dat'
    os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
    print("Final Result", file=open(final_result_path, 'w'))
    
    data_shape, train_past_data, train_future_data, train_data_labels, train_data_weights, \
        test_past_data, test_future_data, test_data_labels, test_data_weights = \
            SPIB_training.data_init(t0, dt, traj_data, traj_labels, traj_weights)
    
    output_path = IB_path + "_d=%d_t=%d_b=%.4f_learn=%f_%s" \
        % (RC_dim, dt, beta, learning_rate, loss_mode)

    IB = SPIB.SPIB(encoder_type, RC_dim, output_dim, data_shape, device, \
                   UpdateLabel, neuron_num1, neuron_num2)
    
    IB.to(device)
    
    # use the training set to initialize the pseudo-inputs
    IB.init_representative_inputs(train_past_data, train_data_labels)

    train_result = False

    train_result = SPIB_training.train(IB, beta, train_past_data, train_future_data, \
                                       train_data_labels, train_data_weights, test_past_data, test_future_data, \
                                           test_data_labels, test_data_weights, learning_rate, lr_scheduler_step_size, lr_scheduler_gamma,\
                                               batch_size, threshold, patience, refinements, output_path, \
                                                   log_interval, device, seed, hsic_config=hsic_config)
    
    if train_result:
        return
    
    SPIB_training.output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_weights, \
                                      test_past_data, test_future_data, test_data_labels, test_data_weights, batch_size, \
                                          output_path, final_result_path, dt, beta, learning_rate, seed,
                                      hsic_config=hsic_config)

    IB.save_traj_results(traj_data, batch_size, output_path, SaveTrajResults, 0, seed)

    if SaveTrajResults and hsic_config.get("DetectTransitionStates", True):
        pred_path = output_path + "_traj0_data_prediction%d.npy" % seed
        pop_path = output_path + "_traj0_state_population%d.npy" % seed
        if os.path.isfile(pred_path):
            SPIB_training.save_and_report_transition_states(
                pred_path, population_path=pop_path if os.path.isfile(pop_path) else None,
                output_prefix=output_path + "_traj0_ts%d" % seed,
                eps_ts=float(hsic_config.get("eps_ts", 0.1)),
                require_cross_state=bool(hsic_config.get("ts_require_cross_state", True)),
                window=int(hsic_config.get("ts_window", 1)),
                log_path=final_result_path)

    if SaveTrajResults and SaveLabelPlot:
        labels_path = output_path + "_traj0_labels%d.npy" % seed
        if os.path.isfile(labels_path):
            traj_np = traj_data.detach().cpu().numpy()
            labels_np = np.load(labels_path)
            fig_name = "%s_HSIC_SPIB_learned_labels_d=%d_t=%d_b=%.4f_traj0_seed%d.png" % (
                loss_mode, RC_dim, dt, beta, seed)
            fig_path = os.path.join(fig_dir, fig_name)
            title = "HSIC-SPIB learned labels (dt=%d)" % dt
            saved, active, pop = plot_state_labels.plot_learned_state_labels(
                traj_np, labels_np, fig_path, title=title)
            print("Saved learned state-label plot: %s (n_states=%d, indices=%s)" % (
                saved, len(active), active.tolist()))

            ts_mask_path = output_path + "_traj0_ts%d_mask.npy" % seed
            if (hsic_config.get("DetectTransitionStates", True)
                    and os.path.isfile(ts_mask_path)):
                ts_mask = np.load(ts_mask_path)
                name_prefix = "%s_HSIC_SPIB_TS_d=%d_t=%d_b=%.4f_traj0_seed%d" % (
                    loss_mode, RC_dim, dt, beta, seed)
                ts_top1, ts_top2 = plot_transition_states.load_decoder_top_pairs(
                    None, ts_mask_path.replace("_mask.npy", "_top2.npy"))
                ts_figs = plot_transition_states.plot_all_ts_figures(
                    traj_np, labels_np, ts_mask, fig_dir, name_prefix,
                    fe_beta=float(hsic_config.get("fe_beta", 3.0)),
                    potential=hsic_config.get("ts_potential", "four_well"),
                    fe_vmax=hsic_config.get("fe_vmax", None),
                    dpi=150,
                    ts_top1=ts_top1,
                    ts_top2=ts_top2,
                    ridge_top_k=hsic_config.get("ts_plot_ridge_top_k"))
                for kind, path, n_ts in ts_figs:
                    print("Saved %s plot: %s (n_TS=%d)" % (kind, path, n_ts))
    
    IB.save_representative_parameters(output_path, seed)


if __name__ == '__main__':
    
    test_model()
    

    
    
