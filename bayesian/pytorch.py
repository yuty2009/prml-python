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
def bayesreg(y, X):
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
    model = BayesLinear(P, 1, prior, dtype=X.dtype).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.08)

    maxsteps = 1500
    for step in range(maxsteps):

        yp, loss_kl = model(X, sample=True)
        # loss_mse = F.mse_loss(yp.squeeze(), y, reduction='sum')
        loss_mse = neg_log_likelihood(yp.squeeze(), y, sigma=1)
        loss = loss_mse + loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics every 100 steps
        if (step + 1) % 10 == 0:
            print("Step [{}/{}], MSE Loss: {:.4f}, "
                  "KL Div: {:.4f} Total loss {:.4f}"
                  .format((step + 1), maxsteps,
                          loss_mse.item(), loss_kl.item(), loss.item()))

    w, b = model.get_weights()
    return w.cpu().detach().numpy(), b.cpu().detach().numpy()


## Bayesian linear regression with ARD prior
# refer to Page 347-348 of PRML book
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# b: P by 1 regression coefficients
# b0: the intercept
def bardreg(y, X):
    N, P = X.shape

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)

    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    prior_rho = nn.Parameter(torch.zeros(P, 1, device=device), requires_grad=True)
    model = BayesLinear(P, 1, prior=prior_rho, dtype=X.dtype).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.08)

    maxsteps = 1500
    for step in range(maxsteps):

        yp, loss_kl = model(X, sample=True)
        # loss_mse = F.mse_loss(yp.squeeze(), y, reduction='sum')
        loss_mse = neg_log_likelihood(yp.squeeze(), y, sigma=1)
        loss = loss_mse + loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics every 100 steps
        if (step + 1) % 10 == 0:
            print("Step [{}/{}], MSE Loss: {:.4f}, "
                  "KL Div: {:.4f} Total loss {:.4f} sum(prior_rho) {:.4f}"
                  .format((step + 1), maxsteps,
                          loss_mse.item(), loss_kl.item(),
                          loss.item(), prior_rho.sum().item()))

    w, b = model.get_weights()
    return w.cpu().detach().numpy(), b.cpu().detach().numpy()


## Bayesian linear regression with grouped ARD prior
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# group: No. of groups or a group id vector
#        e.g. group = [1 1 1 2 2 2 3 3 3 4 4 4]' indicates that there are
#        4 group with 3 members in each
# b: P by 1 regression coefficients
# b0: the intercept
def bgardreg(y, X, group):
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
        group = np.ceil(np.arange(P)/PG).astype(int)
    groupid = np.unique(group)
    NG = len(groupid)

    prior_rho_list = []
    for g in range(NG):
        index_ig = np.argwhere(group == groupid[g])
        prior_rho_ig = nn.Parameter(torch.zeros(len(index_ig), 1, device=device),
                                    requires_grad=True)
        prior_rho_list.append(prior_rho_ig)
    prior_rho = torch.cat(prior_rho_list)

    model = BayesLinear(P, 1, prior=prior_rho, dtype=X.dtype).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.08)

    maxsteps = 1500
    for step in range(maxsteps):

        yp, loss_kl = model(X, sample=True)
        # loss_mse = F.mse_loss(yp.squeeze(), y, reduction='sum')
        loss_mse = neg_log_likelihood(yp.squeeze(), y, sigma=1)
        loss = loss_mse + loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics every 100 steps
        if (step + 1) % 10 == 0:
            print("Step [{}/{}], MSE Loss: {:.4f}, "
                  "KL Div: {:.4f} Total loss {:.4f} sum(prior_rho) {:.4f}"
                  .format((step + 1), maxsteps,
                          loss_mse.item(), loss_kl.item(),
                          loss.item(), prior_rho.sum().item()))

    w, b = model.get_weights()
    return w.cpu().detach().numpy(), b.cpu().detach().numpy()

if __name__ == "__main__":
    pass
