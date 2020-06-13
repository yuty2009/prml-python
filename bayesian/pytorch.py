# -*- coding: utf-8 -*-

import numpy as np
import torch.optim as optim
# import torch.distributions as tfp
from deeplearning.bnn.bayeslayers import *


## Bayesian linear regression
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# w: P by 1 regression coefficients
# b: the intercept
def bayesreg(y, X, sigma=1.0):
    N, P = X.shape

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)

    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    # prior = tfp.Normal(loc=0.0, scale=1.0)
    # prior = LaplacePrior(mu=0, b=1.0)
    prior = GaussPrior(mu=0, sigma=1.0)
    # prior = GaussMixturePrior(mus=[0, 0], sigmas=[1.5, 0.1], pis=[0.5, 0.5])

    w_mu = nn.Parameter(torch.empty(P, 1).normal_(0, 0.5).to(device))
    b_mu = nn.Parameter(torch.empty(1).normal_(0, 0.5).to(device))
    w_rho = nn.Parameter(-3 * torch.ones_like(w_mu).to(device))
    b_rho = nn.Parameter(-3 * torch.ones_like(b_mu).to(device))

    parameters = [w_mu, w_rho, b_mu, b_rho]
    optimizer = optim.Adam(parameters, lr=0.08)

    maxsteps = 1500
    for step in range(maxsteps):

        w, b = sample_weights(w_mu, w_rho, b_mu, b_rho)
        yp = torch.mm(X, w) + b.expand(X.shape[0], -1)

        w_sigma = 1e-6 + F.softplus(w_rho, beta=1, threshold=20)
        b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)

        loss_kl = kl_loss(prior, w, w_mu, w_sigma) + kl_loss(prior, b, b_mu, b_sigma)
        loss_mse = neg_log_likelihood(yp.squeeze(), y, sigma=sigma)
        loss = loss_mse + loss_kl

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # print statistics every 100 steps
        if (step + 1) % 10 == 0:
            print("Step [{}/{}], MSE Loss: {:.4f}, "
                  "KL Div: {:.4f} Total loss {:.4f}"
                  .format((step + 1), maxsteps,
                          loss_mse.item(), loss_kl.item(), loss.item()))

    return w_mu.cpu().detach().numpy(), b_mu.cpu().detach().numpy()


## Bayesian linear regression with ARD prior
# refer to Page 347-348 of PRML book
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# b: P by 1 regression coefficients
# b0: the intercept
def bardreg(y, X, sigma=1.0):
    N, P = X.shape

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)

    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    prior_w_rho = nn.Parameter(torch.zeros(P, 1, device=device))
    prior_b = GaussPrior(0, 0.5)

    w_mu = nn.Parameter(torch.empty(P, 1).normal_(0, 0.5).to(device))
    b_mu = nn.Parameter(torch.empty(1).normal_(0, 0.5).to(device))
    w_rho = nn.Parameter(-3 * torch.ones_like(w_mu).to(device))
    b_rho = nn.Parameter(-3 * torch.ones_like(b_mu).to(device))

    parameters = [w_mu, w_rho, b_mu, b_rho, prior_w_rho]
    optimizer = optim.Adam(parameters, lr=0.08)

    maxsteps = 1500
    for step in range(maxsteps):

        w, b = sample_weights(w_mu, w_rho, b_mu, b_rho)
        yp = torch.mm(X, w) + b.expand(X.shape[0], -1)

        prior_w_sigma = 1e-6 + F.softplus(prior_w_rho, beta=1, threshold=20)
        prior_w = GaussPrior(mu=0, sigma=prior_w_sigma)
        w_sigma = 1e-6 + F.softplus(w_rho, beta=1, threshold=20)
        b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)

        loss_kl = kl_loss(prior_w, w, w_mu, w_sigma) + \
                  kl_loss(prior_b, b, b_mu, b_sigma)
        loss_mse = neg_log_likelihood(yp.squeeze(), y, sigma=sigma)
        loss = loss_mse + loss_kl

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # print statistics every 100 steps
        if (step + 1) % 10 == 0:
            print("Step [{}/{}], MSE Loss: {:.4f}, "
                  "KL Div: {:.4f} Total loss {:.4f} sum(prior_w_rho) {:.4f}"
                  .format((step + 1), maxsteps,
                          loss_mse.item(), loss_kl.item(),
                          loss.item(), prior_w_rho.sum().item()))

    return w_mu.cpu().detach().numpy(), b_mu.cpu().detach().numpy()


## Bayesian linear regression with grouped ARD prior
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# group: No. of groups or a group id vector
#        e.g. group = [1 1 1 2 2 2 3 3 3 4 4 4]' indicates that there are
#        4 group with 3 members in each
# b: P by 1 regression coefficients
# b0: the intercept
def bgardreg(y, X, group, sigma=1.0):
    N, P = X.shape

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)

    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    if np.size(group) == 1:
        PG = np.floor(P/group).astype(int)  # number of feature per-group
        group = np.ceil(np.arange(P)/PG).astype(int) + 1
    groupid = np.unique(group)
    NG = len(groupid)

    prior_w_rho = nn.Parameter(torch.zeros(NG, 1, device=device))
    prior_b = GaussPrior(0, 0.5)

    w_mu = nn.Parameter(torch.empty(P, 1).normal_(0, 0.5).to(device))
    b_mu = nn.Parameter(torch.empty(1).normal_(0, 0.5).to(device))
    w_rho = nn.Parameter(-3 * torch.ones_like(w_mu).to(device))
    b_rho = nn.Parameter(-3 * torch.ones_like(b_mu).to(device))

    parameters = [w_mu, w_rho, b_mu, b_rho, prior_w_rho]
    optimizer = optim.Adam(parameters, lr=0.08)

    maxsteps = 1500
    for step in range(maxsteps):

        w, b = sample_weights(w_mu, w_rho, b_mu, b_rho)
        yp = torch.mm(X, w) + b.expand(X.shape[0], -1)

        prior_w_rho_full = []
        for g in range(NG):
            index_ig = np.argwhere(group == groupid[g])
            prior_w_rho_full.append(prior_w_rho[g].expand(len(index_ig), 1))
        prior_w_rho_full = torch.cat(prior_w_rho_full)
        prior_w_sigma = 1e-6 + F.softplus(prior_w_rho_full, beta=1, threshold=20)
        prior_w = GaussPrior(mu=0, sigma=prior_w_sigma)
        w_sigma = 1e-6 + F.softplus(w_rho, beta=1, threshold=20)
        b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)

        loss_kl = kl_loss(prior_w, w, w_mu, w_sigma) + \
                  kl_loss(prior_b, b, b_mu, b_sigma)
        loss_mse = neg_log_likelihood(yp.squeeze(), y, sigma=sigma)
        loss = loss_mse + loss_kl

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # print statistics every 100 steps
        if (step + 1) % 10 == 0:
            print("Step [{}/{}], MSE Loss: {:.4f}, "
                  "KL Div: {:.4f} Total loss {:.4f} sum(prior_w_rho) {:.4f}"
                  .format((step + 1), maxsteps,
                          loss_mse.item(), loss_kl.item(),
                          loss.item(), prior_w_rho.sum().item()))

    return w_mu.cpu().detach().numpy(), b_mu.cpu().detach().numpy()


def neg_log_likelihood(y_obs, y_pred, sigma=1.0):
    dist = GaussPrior(mu=y_pred, sigma=sigma)
    return -dist.log_prob(y_obs).sum()


def kl_loss(prior, w, mu, sigma):
    variational_dist = GaussPrior(mu, sigma)
    # variational_dist = tfp.Normal(mu, sigma)
    return torch.sum(variational_dist.log_prob(w) - prior.log_prob(w))


def sample_weights(w_mu, w_rho, b_mu, b_rho):
    """Quick method for sampling weights and exporting weights"""
    W_eps = w_mu.data.new(w_mu.size()).normal_()
    w_sigma = 1e-6 + F.softplus(w_rho, beta=1, threshold=20)
    W = w_mu + 1 * w_sigma * W_eps

    if b_mu is not None:
        b_eps = b_mu.data.new(b_mu.size()).normal_()
        b_sigma = 1e-6 + F.softplus(b_rho, beta=1, threshold=20)
        b = b_mu + 1 * b_sigma * b_eps
    else:
        b = None

    return W, b


if __name__ == "__main__":
    pass
