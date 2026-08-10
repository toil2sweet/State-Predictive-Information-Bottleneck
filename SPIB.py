"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198

HSIC-SPIB+: decoder head pruning (2024-style update_model), optional
isotropic encoder variance, and optional input DataNormalize.
"""
import torch
from torch import nn
import numpy as np
import os
import torch.nn.functional as F


class DataNormalize(nn.Module):
    """Fixed affine normalization (mean/std), aligned with 2024 spib_msm."""

    def __init__(self, mean, std):
        super(DataNormalize, self).__init__()
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std = np.asarray(std, dtype=np.float32).reshape(-1)
        std = np.where(std < 1e-8, 1.0, std)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean[None, :]) / self.std[None, :]


# --------------------
# Model
# --------------------

class SPIB(nn.Module):

    def __init__(self, encoder_type, z_dim, output_dim, data_shape, device, UpdateLabel=False,
                 neuron_num1=128, neuron_num2=128, data_transform=None,
                 encoder_var_mode='input_dependent'):
        
        super(SPIB, self).__init__()
        if encoder_type == 'Nonlinear':
            self.encoder_type = 'Nonlinear'
        else:
            self.encoder_type = 'Linear'

        self.z_dim = z_dim
        self.output_dim = output_dim
        
        self.neuron_num1 = neuron_num1
        self.neuron_num2 = neuron_num2
        
        self.data_shape = data_shape
        self.data_transform = data_transform
        
        self.UpdateLabel = UpdateLabel
        
        self.eps = 1e-10
        self.device = device

        if encoder_var_mode not in ('input_dependent', 'isotropic'):
            raise ValueError("encoder_var_mode must be 'input_dependent' or 'isotropic'")
        self.encoder_var_mode = encoder_var_mode

        # [refinement_id, epochs_used, n_states]
        self.convergence_history = []

        # representative-inputs
        self.representative_dim = output_dim

        # torch buffer, these variables will not be trained
        self.representative_inputs = torch.eye(
            self.output_dim, np.prod(self.data_shape), device=device, requires_grad=False)
        
        # create an idle input for calling representative-weights
        self.idle_input = torch.eye(
            self.output_dim, self.output_dim, device=device, requires_grad=False)

        # representative weights
        self.representative_weights = nn.Sequential(
            nn.Linear(self.output_dim, 1, bias=False),
            nn.Softmax(dim=0))
        
        self.encoder = self._encoder_init()

        if self.encoder_type == 'Nonlinear': 
            self.encoder_mean = nn.Linear(self.neuron_num1, self.z_dim)
        else:
            self.encoder_mean = nn.Linear(np.prod(self.data_shape), self.z_dim)
        
        # Note: encoder_type = 'Linear' only means that z_mean is a linear combination of the input OPs.
        if self.encoder_var_mode == 'isotropic':
            # 2024-style: position-independent trainable log-variance
            self.encoder_logvar = nn.Parameter(torch.tensor([0.0]))
        else:
            # 2021-style: input-dependent log_var in [-10, 0]
            self.encoder_logvar = nn.Sequential(
                nn.Linear(self.neuron_num1, self.z_dim),
                nn.Sigmoid())
        
        self.decoder = self._decoder_body_init()
        self.decoder_output = nn.Sequential(
            nn.Linear(self.neuron_num2, self.output_dim),
            nn.LogSoftmax(dim=1))
        
    def _encoder_init(self):
        modules = []
        if self.data_transform is not None:
            modules += [self.data_transform]
        modules += [nn.Linear(np.prod(self.data_shape), self.neuron_num1)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num1, self.neuron_num1)]
            modules += [nn.ReLU()]
        return nn.Sequential(*modules)
    
    def _decoder_body_init(self):
        # Hidden MLP before the state LogSoftmax head (prunable separately)
        modules = [nn.Linear(self.z_dim, self.neuron_num2)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num2, self.neuron_num2)]
            modules += [nn.ReLU()]
        return nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu
    
    def encode(self, inputs):
        enc = self.encoder(inputs)
        
        if self.encoder_type == 'Nonlinear': 
            z_mean = self.encoder_mean(enc)
        else:
            # Linear mean still uses raw (normalized) inputs; skip data_transform path for mean
            if self.data_transform is not None:
                z_mean = self.encoder_mean(self.data_transform(inputs))
            else:
                z_mean = self.encoder_mean(inputs)

        if self.encoder_var_mode == 'isotropic':
            z_logvar = self.encoder_logvar.expand(z_mean.size(0), self.z_dim)
        else:
            z_logvar = -10 * self.encoder_logvar(enc)
        
        return z_mean, z_logvar

    def decode(self, z):
        return self.decoder_output(self.decoder(z))
    
    def forward(self, data, decoder_on_mean=False):
        inputs = torch.flatten(data, start_dim=1)
        
        z_mean, z_logvar = self.encode(inputs)
        
        z_sample = self.reparameterize(z_mean, z_logvar)
        
        # HSIC-SPIB can decode from z_mean for more stable CE + HSIC training;
        # original SPIB behavior (decoder_on_mean=False) is unchanged.
        if decoder_on_mean:
            outputs = self.decode(z_mean)
        else:
            outputs = self.decode(z_sample)
        
        return outputs, z_sample, z_mean, z_logvar
    
    def log_p(self, z, sum_up=True):
        # get representative_z - representative_dim * z_dim
        representative_z_mean, representative_z_logvar = self.get_representative_z()
        # get representative weights - representative_dim * 1
        w = self.representative_weights(self.idle_input)
        
        # expand z - batch_size * z_dim
        z_expand = z.unsqueeze(1)
        
        representative_mean = representative_z_mean.unsqueeze(0)
        representative_logvar = representative_z_logvar.unsqueeze(0)
        
        # representative log_q
        representative_log_q = -0.5 * torch.sum(
            representative_logvar + torch.pow(z_expand - representative_mean, 2)
            / torch.exp(representative_logvar), dim=2)
        
        if sum_up:
            log_p = torch.sum(torch.log(torch.exp(representative_log_q) @ w + self.eps), dim=1)
        else:
            log_p = torch.log(torch.exp(representative_log_q) * w.T + self.eps)
            
        return log_p
        
    # the prior
    def get_representative_z(self):
        X = self.representative_inputs
        representative_z_mean, representative_z_logvar = self.encode(X)
        return representative_z_mean, representative_z_logvar

    def reset_representative(self, representative_inputs):
        self.representative_dim = representative_inputs.shape[0]
        
        self.idle_input = torch.eye(
            self.representative_dim, self.representative_dim, device=self.device, requires_grad=False)

        self.representative_weights = nn.Sequential(
            nn.Linear(self.representative_dim, 1, bias=False),
            nn.Softmax(dim=0))
        self.representative_weights[0].weight = nn.Parameter(
            torch.ones([1, self.representative_dim], device=self.device))
        
        self.representative_inputs = representative_inputs.clone().detach()
        
    @torch.no_grad()
    def init_representative_inputs(self, inputs, labels):
        state_population = labels.sum(dim=0).cpu()
        
        representative_inputs = []
        
        for i in range(state_population.shape[-1]):
            if state_population[i] > 0:
                index = np.random.randint(0, int(state_population[i].item()))
                representative_inputs += [inputs[labels[:, i].bool()][index].reshape(1, -1)]
            else:
                index = np.random.randint(0, inputs.shape[0])
                representative_inputs += [inputs[index].reshape(1, -1)]
        
        representative_inputs = torch.cat(representative_inputs, dim=0)
        self.reset_representative(representative_inputs.to(self.device))
            
        return representative_inputs

    @torch.no_grad()
    def estimatate_representative_inputs(self, inputs, bias, batch_size, labels=None):
        """Pick one representative input per active state (closest to state center in z)."""
        mean_rep = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)
            z_mean, z_logvar = self.encode(batch_inputs)
            mean_rep += [z_mean]
        
        mean_rep = torch.cat(mean_rep, dim=0)

        if labels is None:
            prediction = []
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size].to(self.device)
                z_mean, _ = self.encode(batch_inputs)
                prediction += [self.decode(z_mean).exp()]
            prediction = torch.cat(prediction, dim=0)
            max_pos = prediction.argmax(1)
            labels = F.one_hot(max_pos, num_classes=self.output_dim)
        
        state_population = labels.sum(dim=0)
        
        representative_inputs = []
        
        for i in range(state_population.shape[-1]):
            if state_population[i] > 0:
                if bias is None:
                    center_z = ((mean_rep[labels[:, i].bool()]).mean(dim=0)).reshape(1, -1)
                else:
                    weights = bias[labels[:, i].bool()].reshape(-1, 1)
                    center_z = ((weights * mean_rep[labels[:, i].bool()]).sum(dim=0) / weights.sum()).reshape(1, -1)
                
                dist = torch.square(mean_rep - center_z).sum(dim=-1)
                index = torch.argmin(dist)
                representative_inputs += [inputs[index].reshape(1, -1)]
        
        if len(representative_inputs) == 0:
            raise ValueError("No non-empty states for representative inputs")

        representative_inputs = torch.cat(representative_inputs, dim=0)
        return representative_inputs

    @torch.no_grad()
    def update_model(self, inputs, input_weights, train_data_labels, test_data_labels,
                     batch_size, eps_rho=0.0, update_representative=True):
        """
        HSIC-SPIB+ / 2024-style refinement:
        prune empty (or rare) states, resize decoder_output, optionally refresh VampPrior.
        Always prunes decoder even when update_representative is False (pure HSIC mode).
        """
        state_population = train_data_labels.sum(dim=0).float() / train_data_labels.shape[0]
        keep = state_population > eps_rho

        if int(keep.sum().item()) < 2:
            raise ValueError("Fewer than 2 states remain after population pruning")

        train_data_labels = train_data_labels[:, keep]
        test_data_labels = test_data_labels[:, keep]

        # Preserve surviving decoder head weights before resizing
        w = self.decoder_output[0].weight[keep]
        b = self.decoder_output[0].bias[keep]

        self.output_dim = int(keep.sum().item())
        self.decoder_output = nn.Sequential(
            nn.Linear(self.neuron_num2, self.output_dim),
            nn.LogSoftmax(dim=1))
        self.decoder_output[0].weight = nn.Parameter(w.to(self.device))
        self.decoder_output[0].bias = nn.Parameter(b.to(self.device))
        self.decoder_output.to(self.device)

        if update_representative:
            representative_inputs = self.estimatate_representative_inputs(
                inputs, input_weights, batch_size, labels=train_data_labels)
            self.reset_representative(representative_inputs.to(self.device))
        else:
            # Keep representative dim consistent with output_dim when KL is unused
            # but still allow log_p / save_representative to run safely.
            if self.representative_inputs.shape[0] != self.output_dim:
                representative_inputs = self.estimatate_representative_inputs(
                    inputs, input_weights, batch_size, labels=train_data_labels)
                self.reset_representative(representative_inputs.to(self.device))

        return train_data_labels, test_data_labels
            
    @torch.no_grad()
    def update_labels(self, inputs, batch_size):
        if self.UpdateLabel:
            labels = []
            
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size].to(self.device)
                z_mean, z_logvar = self.encode(batch_inputs)
                log_prediction = self.decode(z_mean)
                labels += [log_prediction.exp()]
            
            labels = torch.cat(labels, dim=0)
            max_pos = labels.argmax(1)
            labels = F.one_hot(max_pos, num_classes=self.output_dim)
            
            return labels
    
    @torch.no_grad()
    def save_representative_parameters(self, path, index=0):
        representative_path = path + '_representative_inputs' + str(index) + '.npy'
        representative_weight_path = path + '_representative_weight' + str(index) + '.npy'
        representative_z_mean_path = path + '_representative_z_mean' + str(index) + '.npy'
        representative_z_logvar_path = path + '_representative_z_logvar' + str(index) + '.npy'
        os.makedirs(os.path.dirname(representative_path), exist_ok=True)
        
        np.save(representative_path, self.representative_inputs.cpu().data.numpy())
        np.save(representative_weight_path, self.representative_weights(self.idle_input).cpu().data.numpy())
        
        representative_z_mean, representative_z_logvar = self.get_representative_z()
        np.save(representative_z_mean_path, representative_z_mean.cpu().data.numpy())
        np.save(representative_z_logvar_path, representative_z_logvar.cpu().data.numpy())
        
    @torch.no_grad()
    def save_traj_results(self, inputs, batch_size, path, SaveTrajResults, traj_index=0, index=1):
        all_prediction = []
        all_z_sample = []
        all_z_mean = []
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)
        
            z_mean, z_logvar = self.encode(batch_inputs)
            z_sample = self.reparameterize(z_mean, z_logvar)
            log_prediction = self.decode(z_mean)
            
            all_prediction += [log_prediction.exp().cpu()]
            all_z_sample += [z_sample.cpu()]
            all_z_mean += [z_mean.cpu()]
            
        all_prediction = torch.cat(all_prediction, dim=0)
        all_z_sample = torch.cat(all_z_sample, dim=0)
        all_z_mean = torch.cat(all_z_mean, dim=0)
        
        max_pos = all_prediction.argmax(1)
        labels = F.one_hot(max_pos, num_classes=self.output_dim)
        
        population = torch.sum(labels, dim=0).float() / len(inputs)
        
        population_path = path + '_traj%d_state_population' % (traj_index) + str(index) + '.npy'
        os.makedirs(os.path.dirname(population_path), exist_ok=True)
        
        np.save(population_path, population.cpu().data.numpy())
        
        self.save_representative_parameters(path, index)

        if self.encoder_type == 'Linear': 
            z_mean_encoder_weight_path = path + '_z_mean_encoder_weight' + str(index) + '.npy'
            z_mean_encoder_bias_path = path + '_z_mean_encoder_bias' + str(index) + '.npy'
            os.makedirs(os.path.dirname(z_mean_encoder_weight_path), exist_ok=True)

            np.save(z_mean_encoder_weight_path, self.encoder_mean.weight.cpu().data.numpy())
            np.save(z_mean_encoder_bias_path, self.encoder_mean.bias.cpu().data.numpy())
            
        if SaveTrajResults:
            label_path = path + '_traj%d_labels' % (traj_index) + str(index) + '.npy'
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            np.save(label_path, labels.cpu().data.numpy())
            
            prediction_path = path + '_traj%d_data_prediction' % (traj_index) + str(index) + '.npy'
            representation_path = path + '_traj%d_representation' % (traj_index) + str(index) + '.npy'
            mean_representation_path = path + '_traj%d_mean_representation' % (traj_index) + str(index) + '.npy'
            
            os.makedirs(os.path.dirname(mean_representation_path), exist_ok=True)
            
            np.save(prediction_path, all_prediction.cpu().data.numpy())
            np.save(representation_path, all_z_sample.cpu().data.numpy())
            np.save(mean_representation_path, all_z_mean.cpu().data.numpy())

            # HSIC-SPIB+: refinement history for state-number vs Δt analysis
            if len(self.convergence_history) > 0:
                hist_path = path + '_convergence_history' + str(index) + '.npy'
                np.save(hist_path, np.asarray(self.convergence_history, dtype=np.float64))
