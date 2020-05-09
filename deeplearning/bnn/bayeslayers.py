# -*- coding: utf-8 -*-
#
# reference: https://github.com/JavierAntoran/Bayesian-Neural-Networks/

import torch
import torch.nn as nn
import torch.nn.functional as F
from .priors import *


class BayesLinear(nn.Module):

    def __init__(self, n_in, n_out, prior):
        super(BayesLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior

        self.W_mu = torch.tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1)
        self.W_rho = torch.tensor(self.n_in, self.n_out).uniform_(-3, -2)
        self.b_mu = torch.tensor(self.n_out).uniform_(-0.1, 0.1)
        self.b_rho = torch.tensor(self.n_out).uniform_(-3, -2)

        self.W_mu = nn.Parameter(self.W_mu, requires_grad=True)
        self.W_rho = nn.Parameter(self.W_rho, requires_grad=True)
        self.b_mu = nn.Parameter(self.b_mu, requires_grad=True)
        self.b_rho = nn.Parameter(self.b_rho, requires_grad=True)

        self.lpw = 0
        self.lqw = 0

    def forward(self, X, sample=False):
        # When testing return MLE of w for quick validation
        if not sample:
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0
        else:
            W, b = self.sample_weights(self.W_mu, self.W_rho, self.b_mu, self.b_rho)
            output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)

            W_std = 1e-6 + F.softplus(self.W_rho, beta=1, threshold=20)
            b_std = 1e-6 + F.softplus(self.b_rho, beta=1, threshold=20)
            lqw = loglike_isotropic_gauss(W, self.W_mu, W_std) \
                  + loglike_isotropic_gauss(b, self.b_mu, b_std)
            lpw = self.prior.loglike(W) + self.prior.loglike(b)
        return output, lqw, lpw

    def sample_weights(self, W_mu, W_rho, b_mu, b_rho):
        """Quick method for sampling weights and exporting weights"""
        W_eps = W_mu.data.new(W_mu.size()).normal_()
        W_std = 1e-6 + F.softplus(W_rho, beta=1, threshold=20)
        W = W_mu + 1 * W_std * W_eps

        if b_mu is not None:
            b_eps = b_mu.data.new(b_mu.size()).normal_()
            b_std = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)
            b = b_mu + 1 * b_std * b_eps
        else:
            b = None

        return W, b
