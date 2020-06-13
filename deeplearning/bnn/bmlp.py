# -*- coding: utf-8 -*-
#
# reference: https://github.com/JavierAntoran/Bayesian-Neural-Networks/

import numpy as np
from deeplearning.bnn.bayeslayers import *


class BayesMLP(nn.Module):
    """Fully-connected Network with Bayes By Backprop"""

    def __init__(self, input_dim, hidden_dims, acts, priors):
        super(BayesMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]
        self.n_hid = len(hidden_dims)

        if acts is None:
            self.acts = []
            for i in range(self.n_hid - 1):
                self.acts.append(nn.ReLU(inplace=True))
            self.acts.append(None)
        elif isinstance(acts, list) or isinstance(acts, tuple):
            self.acts = acts
        else:
            self.acts = []
            for i in range(self.n_hid):
                self.acts.append(acts)

        if priors is None:
            self.priors = []
            for i in range(self.n_hid):
                self.priors.append(GaussPrior(0, 1.0))
        elif isinstance(priors, list) or isinstance(priors, tuple):
            self.priors = priors
        else:
            self.priors = []
            for i in range(self.n_hid):
                self.priors.append(priors)

        in_dims = [input_dim] + hidden_dims[:-1]
        out_dims = hidden_dims
        self.layers = nn.ModuleList()
        for i in range(self.n_hid):
            if self.priors[i] is None:
                self.layers.append(nn.Linear(in_dims[i], out_dims[i]))
            else:
                self.layers.append(BayesLinear(in_dims[i], out_dims[i], self.priors[i]))

    def forward(self, X, sample=False):

        X = X.view(-1, self.input_dim)  # view(batch_size, input_dim)

        loss_kl = 0
        for layer, act, prior in zip(self.layers, self.acts, self.priors):
            if prior is None:
                X, loss_kl_1 = layer(X), 0
            else:
                X, loss_kl_1 = layer(X, sample)
            if act is not None:
                X = act(X)
            loss_kl = loss_kl + loss_kl_1

        return X, loss_kl

    def predict_mcmc(self, X, n_samples):

        predictions = X.data.new(n_samples, X.shape[0], self.output_dim)
        loss_kl = np.zeros(n_samples)

        for i in range(n_samples):
            y, loss_kl_1 = self.forward(X, sample=True)
            predictions[i] = y
            loss_kl[i] = loss_kl_1

        return torch.mean(predictions, dim=0), loss_kl


if __name__ == "__main__":

    import numpy as np
    import torch.optim as optim
    import matplotlib.pyplot as plt

    def f(x, sigma):
        epsilon = np.random.randn(*x.shape) * sigma
        return 10 * np.sin(2 * np.pi * (x)) + epsilon

    train_size = 32
    noise = 1.0

    X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1).astype(np.float32)
    y = f(X, sigma=noise)
    y_true = f(X, sigma=0.0)

    plt.scatter(X, y, marker='+', label='Training data')
    plt.plot(X, y_true, label='Truth')
    plt.title('Noisy training data and ground truth')
    plt.legend()
    plt.show()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    torch.manual_seed(42)
    if cuda:
        torch.cuda.manual_seed(42)

    X_ = torch.tensor(X, device=device)
    y_ = torch.tensor(y, device=device)

    input_dim = X_.shape[-1]
    hidden_dims = [32, 16, 1]
    mlp = BayesMLP(input_dim=input_dim,
                   hidden_dims=hidden_dims,
                   acts=[nn.ReLU(inplace=True),
                         nn.ReLU(inplace=True),
                         None],
                   # priors=LaplacePrior(mu=0, b=1.0),
                   priors=GaussPrior(mu=0, sigma=1.0),
                   # priors=GaussMixturePrior(mus=[0, 0], sigmas=[1.5, 0.1], pis=[0.5, 0.5]),
                   # priors=['gard', 'gard', 'gard']
                   ).to(device)

    optimizer = optim.Adam(mlp.parameters(), lr=0.08)

    epochs = 1500
    weight_kl = 1.0
    for epoch in range(epochs):

        yp, loss_kl = mlp(X_, sample=True)
        loss_ce = neg_log_likelihood(yp, y_)
        loss = loss_ce + weight_kl * loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics every 100 steps
        if (epoch + 1) % 10 == 0:
            print("Epoch [{}/{}], Likelihood Loss: {:.4f}, "
                  "KL Div: {:.4f} Total loss {:.4f}"
                  .format(epoch + 1, epochs, loss_ce.item(), loss_kl.item(), loss.item()))


    with torch.no_grad():

        X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1).astype(np.float32)
        y_pred_list = []

        X_test_ = torch.tensor(X_test, device=device)
        for i in range(500):
            y_pred, _ = mlp(X_test_, sample=True)
            y_pred_list.append(y_pred.cpu().detach().numpy())

        y_preds = np.concatenate(y_pred_list, axis=1)

        y_mean = np.mean(y_preds, axis=1)
        y_sigma = np.std(y_preds, axis=1)

        plt.plot(X_test, y_mean, 'r-', label='Predictive mean')
        plt.scatter(X, y, marker='+', label='Training data')
        plt.fill_between(X_test.ravel(),
                         y_mean + 2 * y_sigma,
                         y_mean - 2 * y_sigma,
                         alpha=0.5, label='Epistemic uncertainty')
        plt.title('Prediction')
        plt.legend()
        plt.show()
