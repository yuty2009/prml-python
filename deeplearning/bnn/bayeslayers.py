# -*- coding: utf-8 -*-
#
# reference: https://github.com/JavierAntoran/Bayesian-Neural-Networks/
#            https://github.com/kumar-shridhar/PyTorch-BayesianCNN

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
# import torch.distributions as tfp
from .priors import *


class BayesLinear(nn.Module):
    """
    Making the prior sigma learnable, thus hyper-paramters could be learned
    from data within the Empirical Bayesian framework.
    i.e., prior_rho is a single value scalar means IID Gaussian prior.
          prior_rho is a vector (the same size with the weights) lead to ARD prior.
          Refer to bayesian/pytorch.py for usages.
    """
    def __init__(self, in_features, out_features, prior=None, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.prior = prior
        if isinstance(prior, BasePrior):
            self.prior_w = prior
            self.prior_b = prior
        elif isinstance(prior, str):
            if prior.lower() == 'ard':
                self.prior_w_rho = nn.Parameter(torch.zeros(out_features, in_features))
                self.prior_b_rho = nn.Parameter(torch.zeros(out_features))
            elif prior.lower() == 'gard' or prior.lower == 'groupard':
                # grouped by input channel which will lead to a feature selector
                self.prior_w_rho = nn.Parameter(torch.zeros(1, in_features))
                self.prior_b_rho = nn.Parameter(torch.zeros(out_features))
        else:
            assert 'Unknown prior'

        self.w_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.w_rho = nn.Parameter(torch.ones_like(self.w_mu))
        if self.use_bias:
            self.b_mu = nn.Parameter(torch.zeros(out_features))
            self.b_rho = nn.Parameter(torch.zeros_like(self.b_mu))

        self.reset_parameters()

    def reset_parameters(self):
        self.w_mu.data.normal_(0, 0.1)
        self.w_rho.data.normal_(-3, 0.1)
        if self.use_bias:
            self.b_mu.data.normal_(0, 0.1)
            self.b_rho.data.normal_(-3, 0.1)

    def forward(self, x, sample=False):
        # When testing return MLE of w for quick validation
        if not self.training and not sample:
            w, b, loss_kl = self.w_mu, self.b_mu, 0
        else:
            w, b = sample_weights(self.w_mu, self.w_rho, self.b_mu, self.b_rho)

            if isinstance(self.prior, str):
                if self.prior.lower() == 'ard':
                    self.prior_w_rho_full = self.prior_w_rho
                elif self.prior.lower() == 'gard' or self.prior.lower == 'groupard':
                    self.prior_w_rho_full = self.prior_w_rho.expand(self.out_features, -1)
                prior_w_sigma = 1e-6 + F.softplus(self.prior_w_rho_full, beta=1, threshold=20)
                self.prior_w = GaussPrior(mu=0, sigma=prior_w_sigma)
                if self.use_bias:
                    prior_b_sigma = 1e-6 + F.softplus(self.prior_b_rho, beta=1, threshold=20)
                    self.prior_b = GaussPrior(mu=0, sigma=prior_b_sigma)

            w_sigma = 1e-6 + F.softplus(self.w_rho, beta=1, threshold=20)
            loss_kl = kl_loss(self.prior_w, w, self.w_mu, w_sigma)
            if self.use_bias:
                b_sigma = 1e-6 + F.softplus(self.b_rho, beta=1, threshold=20)
                loss_kl += kl_loss(self.prior_b, b, self.b_mu, b_sigma)

        return F.linear(x, w, b), loss_kl


class BayesConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 prior=None, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(BayesConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias

        self.prior = prior
        if isinstance(prior, BasePrior):
            self.prior_w = prior
            self.prior_b = prior
        elif isinstance(prior, str):
            if prior.lower() == 'ard':
                self.prior_w_rho = nn.Parameter(
                    torch.zeros(out_channels, in_channels // groups, *kernel_size))
                self.prior_b_rho = nn.Parameter(torch.zeros(out_channels))
            elif prior.lower() == 'gard' or prior.lower == 'groupard':
                # grouped by input channel which will lead to a feature selector
                self.prior_w_rho = nn.Parameter(torch.zeros(1, in_channels // groups))
                self.prior_b_rho = nn.Parameter(torch.zeros(out_channels))
        else:
            assert 'Unknown prior'

        self.w_mu = nn.Parameter(
            torch.zeros(out_channels, in_channels // groups, *kernel_size))
        self.w_rho = nn.Parameter(torch.zeros_like(self.w_mu))
        if self.use_bias:
            self.b_mu = nn.Parameter(torch.zeros(out_channels))
            self.b_rho = nn.Parameter(torch.zeros_like(self.b_mu))

        self.reset_parmeters()

    def reset_parmeters(self):
        self.w_mu.data.normal_(0, 0.1)
        self.w_rho.data.normal_(-3, 0.1)
        if self.use_bias:
            self.b_mu.data.normal_(0, 0.1)
            self.b_rho.data.normal_(-3, 0.1)

    def forward(self, x, sample=False):
        # When testing return MLE of w for quick validation
        if not self.training and not sample:
            w, b, loss_kl = self.w_mu, self.b_mu, 0
        else:
            w, b = sample_weights(self.w_mu, self.w_rho, self.b_mu, self.b_rho)

            if isinstance(self.prior, str):
                if self.prior.lower() == 'ard':
                    self.prior_w_rho_full = self.prior_w_rho
                elif self.prior.lower() == 'gard' or self.prior.lower == 'groupard':
                    self.prior_w_rho_full = self.prior_w_rho.expand_as(self.w_mu)
                prior_w_sigma = 1e-6 + F.softplus(self.prior_w_rho_full, beta=1, threshold=20)
                self.prior_w = GaussPrior(mu=0, sigma=prior_w_sigma)
                if self.use_bias:
                    prior_b_sigma = 1e-6 + F.softplus(self.prior_b_rho, beta=1, threshold=20)
                    self.prior_b = GaussPrior(mu=0, sigma=prior_b_sigma)

            w_sigma = 1e-6 + F.softplus(self.w_rho, beta=1, threshold=20)
            loss_kl = kl_loss(self.prior_w, w, self.w_mu, w_sigma)
            if self.use_bias:
                b_sigma = 1e-6 + F.softplus(self.b_rho, beta=1, threshold=20)
                loss_kl += kl_loss(self.prior_b, b, self.b_mu, b_sigma)

        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups), loss_kl


def neg_log_likelihood(y_obs, y_pred, sigma=1.0):
    dist = GaussPrior(mu=y_pred, sigma=sigma)
    return -dist.log_prob(y_obs).sum()


def kl_loss(prior, w, mu, sigma):
    variational_dist = GaussPrior(mu, sigma)
    # variational_dist = tfp.Normal(mu, sigma)
    return torch.sum(variational_dist.log_prob(w) - prior.log_prob(w))


def kl_loss_gauss(p_mu, p_sigma, q_mu, q_sigma):
    KLD = 0.5 * (2 * torch.log(q_sigma / p_sigma) - 1
                 + (q_sigma / p_sigma).pow(2)
                 + ((p_mu - q_mu) / p_sigma).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return KLD


def sample_weights(w_mu, w_rho, b_mu, b_rho):
    """Quick method for sampling weights and exporting weights"""
    w_eps = torch.zeros_like(w_mu).normal_()
    w_sigma = 1e-6 + F.softplus(w_rho, beta=1, threshold=20)
    w = w_mu + w_sigma * w_eps

    if b_mu is not None:
        b_eps = torch.zeros_like(b_mu).normal_()
        b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)
        b = b_mu + b_sigma * b_eps
    else:
        b = None

    return w, b
