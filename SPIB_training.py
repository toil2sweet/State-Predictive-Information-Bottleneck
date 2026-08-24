"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198

HSIC-SPIB extension: optional HSIC bottleneck terms while retaining the
SPIB decoder, label refinement, state-number estimation, and
decoder-K_i transition-state identification.
"""
import torch
import numpy as np
import time
import os
import warnings

import hsic_utils

# Data Processing
# ------------------------------------------------------------------------------

def data_init(t0, dt, traj_data, traj_label, traj_weights):
    assert len(traj_data)==len(traj_label)
    
    # skip the first t0 data
    past_data = traj_data[t0:(len(traj_data)-dt)]
    future_data = traj_data[(t0+dt):len(traj_data)]
    label = traj_label[(t0+dt):len(traj_data)]
    
    # data shape
    data_shape = past_data.shape[1:]
    
    n_data = len(past_data)
    
    # 90% random test/train split
    p = np.random.permutation(n_data)
    past_data = past_data[p]
    future_data = future_data[p]
    label = label[p]
    
    past_data_train = past_data[0: (9 * n_data) // 10]
    past_data_test = past_data[(9 * n_data) // 10:]
    
    future_data_train = future_data[0: (9 * n_data) // 10]
    future_data_test = future_data[(9 * n_data) // 10:]
    
    label_train = label[0: (9 * n_data) // 10]
    label_test = label[(9 * n_data) // 10:]
    
    if traj_weights != None:
        assert len(traj_data)==len(traj_weights)
        weights = traj_weights[t0:(len(traj_data)-dt)]
        weights = weights[p]
        weights_train = weights[0: (9 * n_data) // 10]
        weights_test = weights[(9 * n_data) // 10:]
    else:
        weights_train = None
        weights_test = None
    
    return data_shape, past_data_train, future_data_train, label_train, weights_train,\
        past_data_test, future_data_test, label_test, weights_test


def default_hsic_config():
    """Default HSIC-SPIB / HSIC-SPIB+ options (including post-hoc TS detection)."""
    return {
        "loss_mode": "original_spib",  # original_spib | hsic_spib | hybrid_spib
        "lambda_y": 0.0,               # weight for -HSIC(Z, Y); 0 avoids CE redundancy by default
        "beta_x": 1.0,                 # weight for +HSIC(Z, X)
        "beta_kl": None,               # if None, use train() beta for hybrid/original
        "kernel_z": "rbf",
        "kernel_x": "rbf",
        "kernel_y": "delta",
        "sigma_z": None,               # None => median heuristic
        "sigma_x": None,
        "sigma_y": None,
        "normalized_hsic": True,
        "decoder_on_mean": True,       # only used when loss_mode != original_spib
        "update_representative": None, # None => True iff KL is used
        "hsic_batch_warn": 4096,
        # HSIC-SPIB+: population pruning threshold and encoder variance mode
        "eps_rho": 0.0,                # prune states with population <= eps_rho
        "encoder_var_mode": "input_dependent",  # input_dependent | isotropic
        # Transition-state identification (decoder K_i margin; post-training)
        # Off by default so plain SPIB configs (no [HSIC-SPIB]) do not auto-plot TS.
        "DetectTransitionStates": False,
        "eps_ts": 0.1,                 # Margin = K_(1)-K_(2) < eps_ts => TS candidate
        "ts_require_cross_state": True,
        "ts_window": 1,
        "fe_beta": 3.0,                # free-energy plot: F=-log(P)/fe_beta
        "fe_vmax": None,               # optional FE color ceiling; Müller default 8 in plotter
        # Must be set explicitly in config: four_well | double_well | muller
        "ts_potential": None,
    }


def _resolve_hsic_config(hsic_config, beta):
    cfg = default_hsic_config()
    if hsic_config:
        cfg.update(hsic_config)
    if cfg["beta_kl"] is None:
        cfg["beta_kl"] = beta
    mode = cfg["loss_mode"]
    if mode not in ("original_spib", "hsic_spib", "hybrid_spib"):
        raise ValueError("Unknown loss_mode: {}".format(mode))
    use_kl = mode in ("original_spib", "hybrid_spib")
    use_hsic = mode in ("hsic_spib", "hybrid_spib")
    if cfg["update_representative"] is None:
        cfg["update_representative"] = use_kl
    cfg["_use_kl"] = use_kl
    cfg["_use_hsic"] = use_hsic
    # original SPIB: keep sampled-z decoder unless user overrides via original path
    if mode == "original_spib":
        cfg["decoder_on_mean"] = False
    return cfg


def count_metastable_states(state_population, eps_rho=0.0):
    """C* = number of states with population > eps_rho (SPIB state-number logic)."""
    active = state_population > eps_rho
    n_states = int(torch.sum(active).item())
    active_indices = torch.nonzero(active, as_tuple=True)[0].tolist()
    return n_states, active_indices


def identify_transition_states(prediction, active_indices=None, eps_ts=0.1,
                               require_cross_state=True, window=1):
    """
    SPIB-style transition-state identification from decoder probabilities K_i.

    Uses only active metastable states. For each frame:
      Margin = K_(1) - K_(2)   (top-1 minus top-2 among active states)
      Balance / TSScore = 1 - Margin
    Two-state case is equivalent to TSScore = 1 - |K_A - K_B|.

    Candidates: Margin < eps_ts. Optionally keep only frames whose hard-label
    neighborhood (within ``window``) contains at least two distinct active states
    (true inter-basin crossing context).

    Parameters
    ----------
    prediction : array-like, shape (N, C)
        Decoder state-transition probabilities K_i(X; Δt).
    active_indices : sequence of int or None
        Indices of metastable states with ρ_i > 0. If None, inferred as
        columns with any positive mass in argmax labels.
    eps_ts : float
        Margin threshold; frames with Margin < eps_ts are TS candidates.
    require_cross_state : bool
        If True, filter candidates by temporal cross-state context.
    window : int
        Half-width (in frames) for the temporal neighborhood check.

    Returns
    -------
    dict with keys:
        n_transition_states, ts_mask, margin, balance, top1, top2,
        n_metastable, active_indices, eps_ts
    """
    K = np.asarray(prediction, dtype=np.float64)
    if K.ndim != 2:
        raise ValueError("prediction must have shape (N, C), got {}".format(K.shape))
    n_frames, n_classes = K.shape

    hard = np.argmax(K, axis=1)
    if active_indices is None:
        pop = np.bincount(hard, minlength=n_classes).astype(np.float64) / max(n_frames, 1)
        active_indices = np.where(pop > 0)[0]
    else:
        active_indices = np.asarray(active_indices, dtype=int)

    n_metastable = int(len(active_indices))
    if n_metastable < 2:
        empty = np.zeros(n_frames, dtype=bool)
        return {
            "n_transition_states": 0,
            "ts_mask": empty,
            "margin": np.ones(n_frames, dtype=np.float64),
            "balance": np.zeros(n_frames, dtype=np.float64),
            "top1": hard,
            "top2": hard.copy(),
            "n_metastable": n_metastable,
            "active_indices": active_indices.tolist(),
            "eps_ts": float(eps_ts),
        }

    K_active = K[:, active_indices]
    # top-2 among active states only
    order = np.argsort(-K_active, axis=1)
    top1_local = order[:, 0]
    top2_local = order[:, 1]
    k1 = K_active[np.arange(n_frames), top1_local]
    k2 = K_active[np.arange(n_frames), top2_local]
    margin = k1 - k2
    balance = 1.0 - margin
    top1 = active_indices[top1_local]
    top2 = active_indices[top2_local]

    ts_mask = margin < float(eps_ts)

    if require_cross_state and window >= 0:
        hard_active = top1  # hard label restricted to active top-1
        cross = np.zeros(n_frames, dtype=bool)
        for t in range(n_frames):
            lo = max(0, t - window)
            hi = min(n_frames, t + window + 1)
            neigh = hard_active[lo:hi]
            if np.unique(neigh).size >= 2:
                cross[t] = True
        ts_mask = ts_mask & cross

    n_ts = int(np.sum(ts_mask))
    return {
        "n_transition_states": n_ts,
        "ts_mask": ts_mask,
        "margin": margin,
        "balance": balance,
        "top1": top1,
        "top2": top2,
        "n_metastable": n_metastable,
        "active_indices": active_indices.tolist(),
        "eps_ts": float(eps_ts),
    }


def save_and_report_transition_states(prediction_path, population_path=None,
                                      output_prefix=None, eps_ts=0.1,
                                      require_cross_state=True, window=1,
                                      log_path=None):
    """
    Load saved decoder predictions, identify transition states, save arrays,
    and print the transition-state count (after metastable state number).
    """
    prediction = np.load(prediction_path)

    active_indices = None
    n_metastable = None
    if population_path is not None and os.path.isfile(population_path):
        population = np.load(population_path)
        active_indices = np.where(population > 0)[0]
        n_metastable = int(len(active_indices))

    result = identify_transition_states(
        prediction, active_indices=active_indices, eps_ts=eps_ts,
        require_cross_state=require_cross_state, window=window)

    if n_metastable is None:
        n_metastable = result["n_metastable"]

    if output_prefix is None:
        # strip trailing .npy if present
        output_prefix = prediction_path
        if output_prefix.endswith(".npy"):
            output_prefix = output_prefix[:-4]
        # replace data_prediction with ts_* stem
        if "data_prediction" in output_prefix:
            output_prefix = output_prefix.replace("data_prediction", "ts")
        else:
            output_prefix = output_prefix + "_ts"

    mask_path = output_prefix + "_mask.npy"
    margin_path = output_prefix + "_margin.npy"
    balance_path = output_prefix + "_balance.npy"
    top2_path = output_prefix + "_top2.npy"
    np.save(mask_path, result["ts_mask"])
    np.save(margin_path, result["margin"])
    np.save(balance_path, result["balance"])
    np.save(top2_path, np.stack([result["top1"], result["top2"]], axis=1))

    msg_lines = [
        "Number of metastable states: %d" % n_metastable,
        "Active state indices: %s" % result["active_indices"],
        "Transition-state identification (decoder K_i): eps_ts=%.4f "
        "require_cross_state=%s window=%d" % (
            eps_ts, require_cross_state, window),
        "Number of transition-state frames: %d (of %d, %.4f%%)" % (
            result["n_transition_states"], len(result["ts_mask"]),
            100.0 * result["n_transition_states"] / max(len(result["ts_mask"]), 1)),
    ]
    for line in msg_lines:
        print(line)
        if log_path is not None:
            print(line, file=open(log_path, 'a'))

    return result


# Loss function
# ------------------------------------------------------------------------------

def calculate_loss(IB, data_inputs, data_targets, data_weights, beta=1.0, hsic_config=None,
                   compute_kl=None):
    """
    Multi-mode loss for SPIB / HSIC-SPIB.

    Returns
    -------
    loss, reconstruction_error, kl_loss, hsic_zx, hsic_zy
    """
    cfg = _resolve_hsic_config(hsic_config, beta)
    decoder_on_mean = bool(cfg["decoder_on_mean"]) and cfg["_use_hsic"]

    outputs, z_sample, z_mean, z_logvar = IB.forward(
        data_inputs, decoder_on_mean=decoder_on_mean)

    if data_weights is None:
        reconstruction_error = torch.mean(torch.sum(-data_targets * outputs, dim=1))
    else:
        reconstruction_error = torch.mean(
            data_weights * torch.sum(-data_targets * outputs, dim=1))

    if compute_kl is None:
        compute_kl = cfg["_use_kl"]
    if compute_kl:
        log_p = IB.log_p(z_sample)
        log_q = -0.5 * torch.sum(z_logvar + torch.pow(z_sample - z_mean, 2)
                                 / torch.exp(z_logvar), dim=1)
        if data_weights is None:
            kl_loss = torch.mean(log_q - log_p)
        else:
            kl_loss = torch.mean(data_weights * (log_q - log_p))
    else:
        kl_loss = outputs.new_zeros(())

    hsic_zx = data_inputs.new_zeros(())
    hsic_zy = data_inputs.new_zeros(())

    if cfg["_use_hsic"]:
        batch_size = data_inputs.size(0)
        if batch_size > cfg["hsic_batch_warn"]:
            warnings.warn(
                "HSIC is O(B^2); current batch_size={} may be slow/memory-heavy.".format(
                    batch_size))

        # Z = mu(X) for stable batch-level dependence estimation
        z_for_hsic = z_mean
        x_flat = torch.flatten(data_inputs, start_dim=1)

        kz, _ = hsic_utils.build_kernel(
            z_for_hsic, kernel_type=cfg["kernel_z"],
            sigma=cfg["sigma_z"], detach_kernel=False)
        kx, _ = hsic_utils.build_kernel(
            x_flat, kernel_type=cfg["kernel_x"],
            sigma=cfg["sigma_x"], detach_kernel=True)
        ky, _ = hsic_utils.build_kernel(
            data_targets, kernel_type=cfg["kernel_y"],
            sigma=cfg["sigma_y"], detach_kernel=True)
        hsic_zx = hsic_utils.hsic_from_grams(kz, kx, normalized=cfg["normalized_hsic"])
        hsic_zy = hsic_utils.hsic_from_grams(kz, ky, normalized=cfg["normalized_hsic"])

    loss = reconstruction_error
    if cfg["_use_kl"]:
        loss = loss + float(cfg["beta_kl"]) * kl_loss
    if cfg["_use_hsic"]:
        # minimize: CE - lambda_y * HSIC(Z,Y) + beta_x * HSIC(Z,X)
        loss = loss - float(cfg["lambda_y"]) * hsic_zy + float(cfg["beta_x"]) * hsic_zx

    return (loss, reconstruction_error.float(), kl_loss.float(),
            hsic_zx.float(), hsic_zy.float())


# Train and test model
# ------------------------------------------------------------------------------

def sample_minibatch(past_data, data_labels, data_weights, indices, device):
    sample_past_data = past_data[indices].to(device)
    sample_data_labels = data_labels[indices].to(device)
    
    if data_weights == None:
        sample_data_weights = None
    else:
        sample_data_weights = data_weights[indices].to(device)
    
    
    return sample_past_data, sample_data_labels, sample_data_weights


def _epoch_sample_count(n_train, batch_size, epoch_sample_size):
    """Number of SGD samples this epoch, rounded down to a full minibatch."""
    n_use = int(n_train)
    if epoch_sample_size and int(epoch_sample_size) > 0:
        n_use = min(n_use, int(epoch_sample_size))
    n_use = (n_use // batch_size) * batch_size
    return n_use


def train(IB, beta, train_past_data, train_future_data, init_train_data_labels, train_data_weights, \
          test_past_data, test_future_data, init_test_data_labels, test_data_weights, \
              learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, batch_size, threshold, patience, refinements, output_path, log_interval, device, index,
              hsic_config=None, epoch_sample_size=0):
    IB.train()
    cfg = _resolve_hsic_config(hsic_config, beta)
    
    step = 0
    start = time.time()
    log_path = output_path + '_train.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    IB_path = output_path + "cpt" + str(index) + "/IB"
    os.makedirs(os.path.dirname(IB_path), exist_ok=True)

    eps_rho = float(cfg.get("eps_rho", 0.0))
    n_train = len(train_past_data)
    n_epoch_samples = _epoch_sample_count(n_train, batch_size, epoch_sample_size)
    if n_epoch_samples < batch_size:
        raise ValueError(
            "Not enough training samples for batch_size=%d (n_train=%d, epoch_sample_size=%s)"
            % (batch_size, n_train, epoch_sample_size))
    print("loss_mode=%s lambda_y=%s beta_x=%s beta_kl=%s normalized_hsic=%s "
          "decoder_on_mean=%s eps_rho=%s encoder_var_mode=%s epoch_sample_size=%d/%d" % (
        cfg["loss_mode"], cfg["lambda_y"], cfg["beta_x"], cfg["beta_kl"],
        cfg["normalized_hsic"], cfg["decoder_on_mean"], eps_rho,
        getattr(IB, "encoder_var_mode", cfg.get("encoder_var_mode", "input_dependent")),
        n_epoch_samples, n_train),
          file=open(log_path, 'a'))
    print("SGD samples/epoch=%d of %d train frames (labels/TS still use all data)"
          % (n_epoch_samples, n_train))
    
    train_data_labels = init_train_data_labels
    test_data_labels = init_test_data_labels

    update_times = 0
    unchanged_epochs = 0
    epoch = 0
    IB.convergence_history = []

    # initial state population (used for state-number convergence)
    state_population0 = torch.sum(train_data_labels,dim=0).float()/train_data_labels.shape[0]
    n_states0, active_idx0 = count_metastable_states(state_population0, eps_rho=eps_rho)
    print("Initial metastable state number: %d indices=%s" % (n_states0, active_idx0))
    print("Initial metastable state number: %d indices=%s" % (n_states0, active_idx0),
          file=open(log_path, 'a'))

    # generate the optimizer and scheduler (joint BP over encoder + decoder)
    optimizer = torch.optim.Adam(IB.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

    while True:
        
        train_permutation = torch.randperm(n_train)[:n_epoch_samples]
        test_permutation = torch.randperm(len(test_past_data))
        
        
        for i in range(0, n_epoch_samples, batch_size):
            step += 1
            
            train_indices = train_permutation[i:i+batch_size]
            
            batch_inputs, batch_outputs, batch_weights = sample_minibatch(train_past_data, train_data_labels, \
                                                                       train_data_weights, train_indices, device)
                    
            loss, reconstruction_error, kl_loss, hsic_zx, hsic_zy = calculate_loss(
                IB, batch_inputs, batch_outputs, batch_weights, beta, hsic_config=cfg)
            
            # Stop if NaN is obtained
            if(torch.isnan(loss).any()):
                return True
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 500 == 0:
                with torch.no_grad():
                    
                    batch_inputs, batch_outputs, batch_weights = sample_minibatch(train_past_data, train_data_labels, \
                                                                               train_data_weights, train_indices, device)
                            
                    loss, reconstruction_error, kl_loss, hsic_zx, hsic_zy = calculate_loss(
                        IB, batch_inputs, batch_outputs, batch_weights, beta, hsic_config=cfg,
                        compute_kl=True)
                    train_time = time.time() - start
            
                    print(
                        "Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f\tHSIC_zx (train) %f\tHSIC_zy (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error, hsic_zx, hsic_zy))
                    print(
                       "Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f\tHSIC_zx (train) %f\tHSIC_zy (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error, hsic_zx, hsic_zy),
                        file=open(log_path, 'a'))
                    j=i%len(test_permutation)
                    
                    
                    
                    test_indices = test_permutation[j:j+batch_size]
                    
                    batch_inputs, batch_outputs, batch_weights = sample_minibatch(test_past_data, test_data_labels, \
                                                                               test_data_weights, test_indices, device)
                    
                    loss, reconstruction_error, kl_loss, hsic_zx, hsic_zy = calculate_loss(
                        IB, batch_inputs, batch_outputs, batch_weights, beta, hsic_config=cfg,
                        compute_kl=True)

                    train_time = time.time() - start
                    print(
                       "Loss (test) %f\tKL loss (test): %f\n"
                       "Reconstruction loss (test) %f\tHSIC_zx (test) %f\tHSIC_zy (test) %f" % (
                           loss, kl_loss, reconstruction_error, hsic_zx, hsic_zy))
                    print(
                       "Loss (test) %f\tKL loss (test): %f\n"
                       "Reconstruction loss (test) %f\tHSIC_zx (test) %f\tHSIC_zy (test) %f" % (
                           loss, kl_loss, reconstruction_error, hsic_zx, hsic_zy),
                        file=open(log_path, 'a'))
        
            if step % log_interval == 0:
                # save model
                torch.save({'step': step,
                            'state_dict': IB.state_dict()},
                           IB_path+ '_%d_cpt.pt'%step)
                torch.save({'optimizer': optimizer.state_dict()},
                           IB_path+ '_%d_optim_cpt.pt'%step) 

        epoch+=1
        
        # State-number path: label refinement still uses decoder probabilities
        # (never HSIC scores), preserving SPIB state pruning / C* estimation.
        new_train_data_labels = IB.update_labels(train_future_data, batch_size)

        # save the state population
        state_population = torch.sum(new_train_data_labels,dim=0).float()/new_train_data_labels.shape[0]
        n_states, active_idx = count_metastable_states(state_population, eps_rho=eps_rho)

        print(state_population)
        print(state_population, file=open(log_path, 'a'))
        print("Metastable state number: %d indices=%s" % (n_states, active_idx))
        print("Metastable state number: %d indices=%s" % (n_states, active_idx),
              file=open(log_path, 'a'))

        # print the state population change
        state_population_change = torch.sqrt(torch.square(state_population-state_population0).sum())
        
        print('State population change=%f'%state_population_change)
        print('State population change=%f'%state_population_change, file=open(log_path, 'a'))

        # update state_population
        state_population0 = state_population

        scheduler.step()
        if scheduler.gamma < 1:
            print("Update lr to %f"%(optimizer.param_groups[0]['lr']))
            print("Update lr to %f"%(optimizer.param_groups[0]['lr']), file=open(log_path, 'a'))

        # check whether the change of the state population is smaller than the threshold
        if state_population_change < threshold:
            unchanged_epochs += 1
            
            if unchanged_epochs > patience:

                # check whether only one state is found
                if n_states < 2:
                    print("Only one metastable state is found!")
                    break

                # Stop only if update_times >= refinements
                if IB.UpdateLabel and update_times < refinements:
                    
                    train_data_labels = new_train_data_labels
                    test_data_labels = IB.update_labels(test_future_data, batch_size)
    
                    update_times+=1
                    print("Update %d\n"%(update_times))
                    print("Update %d\n"%(update_times), file=open(log_path, 'a'))

                    # HSIC-SPIB+: always prune empty/rare states and resize decoder head
                    # (even in pure hsic_spib without KL / VampPrior).
                    train_data_labels, test_data_labels = IB.update_model(
                        train_past_data, train_data_weights,
                        train_data_labels, test_data_labels, batch_size,
                        eps_rho=eps_rho,
                        update_representative=bool(cfg["update_representative"]))

                    IB.convergence_history.append(
                        [update_times, epoch, IB.output_dim])
                    print("After prune: output_dim=%d convergence_history=%s" % (
                        IB.output_dim, IB.convergence_history),
                          file=open(log_path, 'a'))
                    print("After prune: output_dim=%d" % IB.output_dim)

                    # reset epoch and unchanged_epochs
                    epoch = 0
                    unchanged_epochs = 0

                    # Recompute population baseline on pruned labels
                    state_population0 = (
                        torch.sum(train_data_labels, dim=0).float() / train_data_labels.shape[0])
    
                    # reset the optimizer and scheduler (new decoder head params)
                    optimizer = torch.optim.Adam(IB.parameters(), lr=learning_rate)

                    scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
                    
                else:
                    break

        else:
            unchanged_epochs = 0

        print("Epoch: %d\n"%(epoch))
        print("Epoch: %d\n"%(epoch), file=open(log_path, 'a'))

    # Final prune so saved labels / decoder dims stay consistent
    if IB.UpdateLabel:
        train_data_labels = IB.update_labels(train_future_data, batch_size)
        test_data_labels = IB.update_labels(test_future_data, batch_size)
        try:
            train_data_labels, test_data_labels = IB.update_model(
                train_past_data, train_data_weights,
                train_data_labels, test_data_labels, batch_size,
                eps_rho=eps_rho,
                update_representative=bool(cfg["update_representative"]))
        except ValueError as exc:
            print("Final prune skipped: %s" % exc)
            print("Final prune skipped: %s" % exc, file=open(log_path, 'a'))

    # output the saving path
    total_training_time = time.time() - start
    print("Total training time: %f" % total_training_time)
    print("Total training time: %f" % total_training_time, file=open(log_path, 'a'))
    # save model
    torch.save({'step': step,
                'state_dict': IB.state_dict()},
               IB_path+ '_%d_cpt.pt'%step)
    torch.save({'optimizer': optimizer.state_dict()},
               IB_path+ '_%d_optim_cpt.pt'%step)
    
    torch.save({'step': step,
                'state_dict': IB.state_dict()},
               IB_path+ '_final_cpt.pt')
    torch.save({'optimizer': optimizer.state_dict()},
               IB_path+ '_final_optim_cpt.pt')

    # Persist refinement history for state-number analysis
    if len(IB.convergence_history) > 0:
        np.save(output_path + '_convergence_history%d.npy' % index,
                np.asarray(IB.convergence_history, dtype=np.float64))

    return False

@torch.no_grad()
def output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_weights, \
                        test_past_data, test_future_data, test_data_labels, test_data_weights, batch_size, output_path, \
                            path, dt, beta, learning_rate, index=0, hsic_config=None):
    
    with torch.no_grad():
        cfg = _resolve_hsic_config(hsic_config, beta)
        final_result_path = output_path + '_final_result' + str(index) + '.npy'
        os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
        
        # label update still based on decoder argmax (state-number path)
        if IB.UpdateLabel:
            train_data_labels = IB.update_labels(train_future_data, batch_size)
            test_data_labels = IB.update_labels(test_future_data, batch_size)
        
        final_result = []
        # output the result
        
        loss = reconstruction_error = kl_loss = hsic_zx = hsic_zy = 0
        
        for i in range(0, len(train_past_data), batch_size):
            batch_inputs, batch_outputs, batch_weights = sample_minibatch(train_past_data, train_data_labels, train_data_weights, \
                                                                       range(i,min(i+batch_size,len(train_past_data))), IB.device)
            loss1, reconstruction_error1, kl_loss1, hsic_zx1, hsic_zy1 = calculate_loss(
                IB, batch_inputs, batch_outputs, batch_weights, beta, hsic_config=cfg)
            loss += loss1*len(batch_inputs)
            reconstruction_error += reconstruction_error1*len(batch_inputs)
            kl_loss += kl_loss1*len(batch_inputs)
            hsic_zx += hsic_zx1*len(batch_inputs)
            hsic_zy += hsic_zy1*len(batch_inputs)
            
        
        # output the result
        loss/=len(train_past_data)
        reconstruction_error/=len(train_past_data)
        kl_loss/=len(train_past_data)
        hsic_zx/=len(train_past_data)
        hsic_zy/=len(train_past_data)
                
        final_result += [loss.data.cpu().numpy(), reconstruction_error.cpu().data.numpy(),
                         kl_loss.cpu().data.numpy(),
                         hsic_zx.cpu().data.numpy(), hsic_zy.cpu().data.numpy()]
        print(
            "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
                    "Reconstruction loss (train) %f\tHSIC_zx (train) %f\tHSIC_zy (train) %f" % (
                index, loss, kl_loss, reconstruction_error, hsic_zx, hsic_zy))
        print(
            "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
                    "Reconstruction loss (train) %f\tHSIC_zx (train) %f\tHSIC_zy (train) %f" % (
                index, loss, kl_loss, reconstruction_error, hsic_zx, hsic_zy),
            file=open(path, 'a'))
    
        loss = reconstruction_error = kl_loss = hsic_zx = hsic_zy = 0
        
        for i in range(0, len(test_past_data), batch_size):
            batch_inputs, batch_outputs, batch_weights = sample_minibatch(test_past_data, test_data_labels, test_data_weights, \
                                                                                         range(i,min(i+batch_size,len(test_past_data))), IB.device)
            loss1, reconstruction_error1, kl_loss1, hsic_zx1, hsic_zy1 = calculate_loss(
                IB, batch_inputs, batch_outputs, batch_weights, beta, hsic_config=cfg)
            loss += loss1*len(batch_inputs)
            reconstruction_error += reconstruction_error1*len(batch_inputs)
            kl_loss += kl_loss1*len(batch_inputs)
            hsic_zx += hsic_zx1*len(batch_inputs)
            hsic_zy += hsic_zy1*len(batch_inputs)
            
        
        # output the result
        loss/=len(test_past_data)
        reconstruction_error/=len(test_past_data)
        kl_loss/=len(test_past_data)
        hsic_zx/=len(test_past_data)
        hsic_zy/=len(test_past_data)
        
        final_result += [loss.cpu().data.numpy(), reconstruction_error.cpu().data.numpy(),
                         kl_loss.cpu().data.numpy(),
                         hsic_zx.cpu().data.numpy(), hsic_zy.cpu().data.numpy()]
        print(
            "Loss (test) %f\tKL loss (test): %f\n"
            "Reconstruction loss (test) %f\tHSIC_zx (test) %f\tHSIC_zy (test) %f"
            % (loss, kl_loss, reconstruction_error, hsic_zx, hsic_zy))
        print( 
            "Loss (test) %f\tKL loss (test): %f\n"
            "Reconstruction loss (test) %f\tHSIC_zx (test) %f\tHSIC_zy (test) %f"
            % (loss, kl_loss, reconstruction_error, hsic_zx, hsic_zy), file=open(path, 'a'))
        
        print("dt: %d\t Beta: %f\t Learning_rate: %f\t loss_mode: %s" % (
            dt, beta, learning_rate, cfg["loss_mode"]))
        print("dt: %d\t Beta: %f\t Learning_rate: %f\t loss_mode: %s" % (
            dt, beta, learning_rate, cfg["loss_mode"]),
              file=open(path, 'a'))

        state_population = torch.sum(train_data_labels, dim=0).float() / train_data_labels.shape[0]
        n_metastable, active_state_indices = count_metastable_states(
            state_population, eps_rho=float(cfg.get("eps_rho", 0.0)))
        print("Number of metastable states: %d" % n_metastable)
        print("State population: %s" % state_population.cpu().numpy())
        print("Active state indices: %s" % active_state_indices)
        print("Number of metastable states: %d" % n_metastable, file=open(path, 'a'))
        print("State population: %s" % state_population.cpu().numpy(), file=open(path, 'a'))
        print("Active state indices: %s" % active_state_indices, file=open(path, 'a'))
        if len(IB.convergence_history) > 0:
            print("convergence_history [refinement, epochs, n_states]: %s" % IB.convergence_history)
            print("convergence_history [refinement, epochs, n_states]: %s" % IB.convergence_history,
                  file=open(path, 'a'))

        # Note: full-trajectory transition-state counts are reported after
        # save_traj_results() via save_and_report_transition_states().
        
        final_result = np.array(final_result)
        np.save(final_result_path, final_result)
