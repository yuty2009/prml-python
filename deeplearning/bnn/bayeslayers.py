# -*- coding: utf-8 -*-
#
# reference: https://github.com/JavierAntoran/Bayesian-Neural-Networks/

import torch
import torch.nn as nn
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
    def __init__(self, n_in, n_out, prior=None, kl_weight=1, dtype=float):
        super(BayesLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if isinstance(prior, BasePrior):
            self.prior = prior
            self.prior_rho = None
        elif isinstance(prior, list) or isinstance(prior, tuple) :
            self.prior = None
            self.prior_rho = torch.cat(prior)
        else:
            self.prior = None
            self.prior_rho=prior
        self.dtype = dtype
        self.kl_weight = kl_weight
        self.init_sigma = 0.5

        self.W_mu = torch.Tensor(self.n_in, self.n_out).normal_(0, self.init_sigma)
        self.b_mu = torch.Tensor(self.n_out).normal_(0, self.init_sigma)
        self.W_rho = torch.zeros_like(self.W_mu)
        self.b_rho = torch.zeros_like(self.b_mu)

        self.W_mu = nn.Parameter(self.W_mu.type(dtype), requires_grad=True)
        self.W_rho = nn.Parameter(self.W_rho.type(dtype), requires_grad=True)
        self.b_mu = nn.Parameter(self.b_mu.type(dtype), requires_grad=True)
        self.b_rho = nn.Parameter(self.b_rho.type(dtype), requires_grad=True)

    def forward(self, X, sample=False):
        # When testing return MLE of w for quick validation
        if not self.training and not sample:
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0
        else:
            if self.prior_rho is not None:
                prior_sigma = 1e-6 + F.softplus(self.prior_rho, beta=1, threshold=20)
                self.prior = GaussPrior(mu=0, sigma=prior_sigma)

            W, b = sample_weights(self.W_mu, self.W_rho, self.b_mu, self.b_rho)
            output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)

            W_sigma = 1e-6 + F.softplus(self.W_rho, beta=1, threshold=20)
            b_sigma = 1e-6 + F.softplus(self.b_rho, beta=1, threshold=20)
            loss_kl = self.kl_loss(W, self.W_mu, W_sigma) + \
                      self.kl_loss(b, self.b_mu, b_sigma)

        return output, loss_kl

    def kl_loss(self, w, mu, sigma):
        variational_dist = GaussPrior(mu, sigma)
        # variational_dist = tfp.Normal(mu, sigma)
        return self.kl_weight * torch.sum(variational_dist.log_prob(w) -
                                      self.prior.log_prob(w))

    def get_weights(self):
        state_dict = self.state_dict()
        W_mu = state_dict['W_mu'].data
        b_mu = state_dict['b_mu'].data
        return W_mu, b_mu

    def get_weight_samples(self, n_samples=100):
        state_dict = self.state_dict()
        W_mu = state_dict['W_mu'].data
        W_rho = state_dict['W_rho'].data
        b_mu = state_dict['b_mu'].data
        b_rho = state_dict['b_rho'].data
        Ws = []
        bs = []
        for i in range(n_samples):
            W, b = sample_weights(W_mu, W_rho, b_mu, b_rho)
            Ws.append(W)
            bs.append(b)
        return Ws, bs


class BayesLinearEx(nn.Module):

    def __init__(self, n_in, n_out, prior_rho=None, kl_weight=1, dtype=float):
        super(BayesLinearEx, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior_rho = prior_rho
        self.dtype = dtype
        self.kl_weight = kl_weight
        self.init_sigma = 0.5

        self.W_mu = torch.Tensor(self.n_in, self.n_out).normal_(0, self.init_sigma)
        self.b_mu = torch.Tensor(self.n_out).normal_(0, self.init_sigma)
        self.W_rho = torch.zeros_like(self.W_mu)
        self.b_rho = torch.zeros_like(self.b_mu)

        self.W_mu = nn.Parameter(self.W_mu.type(dtype), requires_grad=True)
        self.W_rho = nn.Parameter(self.W_rho.type(dtype), requires_grad=True)
        self.b_mu = nn.Parameter(self.b_mu.type(dtype), requires_grad=True)
        self.b_rho = nn.Parameter(self.b_rho.type(dtype), requires_grad=True)

    def forward(self, X, sample=False):
        # When testing return MLE of w for quick validation
        if not self.training and not sample:
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0
        else:
            prior_sigma = 1e-6 + F.softplus(self.prior_rho, beta=1, threshold=20)
            self.prior = GaussPrior(mu=0, sigma=prior_sigma)

            W, b = sample_weights(self.W_mu, self.W_rho, self.b_mu, self.b_rho)
            output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)

            W_sigma = 1e-6 + F.softplus(self.W_rho, beta=1, threshold=20)
            b_sigma = 1e-6 + F.softplus(self.b_rho, beta=1, threshold=20)
            loss_kl = self.kl_loss(W, self.W_mu, W_sigma) + \
                      self.kl_loss(b, self.b_mu, b_sigma)

        return output, loss_kl

    def kl_loss(self, w, mu, sigma):
        variational_dist = GaussPrior(mu, sigma)
        # variational_dist = tfp.Normal(mu, sigma)
        return self.kl_weight * torch.sum(variational_dist.log_prob(w) -
                                      self.prior.log_prob(w))

    def get_weights(self):
        state_dict = self.state_dict()
        W_mu = state_dict['W_mu'].data
        b_mu = state_dict['b_mu'].data
        return W_mu, b_mu

    def get_weight_samples(self, n_samples=100):
        state_dict = self.state_dict()
        W_mu = state_dict['W_mu'].data
        W_rho = state_dict['W_rho'].data
        b_mu = state_dict['b_mu'].data
        b_rho = state_dict['b_rho'].data
        Ws = []
        bs = []
        for i in range(n_samples):
            W, b = sample_weights(W_mu, W_rho, b_mu, b_rho)
            Ws.append(W)
            bs.append(b)
        return Ws, bs


def neg_log_likelihood(y_obs, y_pred, sigma=1.0):
    dist = GaussPrior(mu=y_pred, sigma=sigma)
    return -dist.log_prob(y_obs).sum()


def kld_cost_gauss(p_mu, p_sigma, q_mu, q_sigma):
    KLD = 0.5 * (2 * torch.log(q_sigma / p_sigma) - 1
                 + (q_sigma / p_sigma).pow(2) + ((p_mu - q_mu) / p_sigma).pow(2)).sum()
    # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return KLD


def sample_weights(W_mu, W_rho, b_mu, b_rho, dtype=float):
    """Quick method for sampling weights and exporting weights"""
    W_eps = W_mu.data.new(W_mu.size()).normal_()
    W_sigma = 1e-6 + F.softplus(W_rho, beta=1, threshold=20)
    W = W_mu + 1 * W_sigma * W_eps

    if b_mu is not None:
        b_eps = b_mu.data.new(b_mu.size()).normal_()
        b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)
        b = b_mu + 1 * b_sigma * b_eps
    else:
        b = None

    return W, b
