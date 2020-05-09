# -*- coding: utf-8 -*-
'''
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
[2] https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim):
        super(VAE, self).__init__()
        # encoder
        self.dense1 = nn.Linear(x_dim, h_dim)
        self.dense2 = nn.Linear(h_dim, z_dim)
        self.dense3 = nn.Linear(h_dim, z_dim)
        # decoder
        self.dense4 = nn.Linear(z_dim, h_dim)
        self.dense5 = nn.Linear(h_dim, x_dim)

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z_sampled = self.reparameterize(z_mean, z_log_var)
        x_decoded = self.decode(z_sampled)
        return x_decoded, z_mean, z_log_var

    def encode(self, x):
        h = F.relu(self.dense1(x))
        z_mean = self.dense2(h)
        z_log_var = self.dense3(h)  # estimate log(sigma**2) actually
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        epsilon = torch.exp(0.5 * z_log_var)
        return z_mean + epsilon * torch.randn_like(epsilon)

    def decode(self, z):
        h = F.relu(self.dense4(z))
        x_decoded = F.sigmoid(self.dense5(h))
        return x_decoded

    def loss(self, inputs, outputs, z_mean, z_log_var):
        loss_recon = F.binary_cross_entropy(outputs, inputs, reduction='sum')
        loss_kl = 0.5 * torch.sum(torch.exp(z_log_var) + torch.pow(z_mean, 2)
                                  - 1. - z_log_var)
        loss_vae = loss_recon + loss_kl
        return loss_vae, loss_recon, loss_kl
