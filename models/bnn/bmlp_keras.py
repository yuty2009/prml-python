# -*- coding: utf-8 -*-
#
# reference: http://krasserm.github.io/2019/03/14/bayesian-neural-networks/

from keras.models import Sequential
from deeplearning.bnn.bayeslayers_keras import *


class BayesMLP:
    def __init__(self, input_dim, hidden_dims, acts, priors, kl_weight=1):
        super(BayesMLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]
        self.n_hid = len(hidden_dims)

        if acts is None:
            self.acts = []
            for i in range(self.n_hid - 1):
                self.acts.append('relu')
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
                self.priors.append(GaussPrior(0, 0.1))
        elif isinstance(priors, list) or isinstance(priors, tuple):
            self.priors = priors
        else:
            self.priors = []
            for i in range(self.n_hid):
                self.priors.append(priors)

        self.model = Sequential()
        for i in range(self.n_hid):
            self.model.add(BayesLinear(units=hidden_dims[i],
                                       prior=self.priors[i],
                                       activation=self.acts[i],
                                       kl_weight=kl_weight))


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt
    from keras import callbacks, optimizers

    def f(x, sigma):
        epsilon = np.random.randn(*x.shape) * sigma
        return 10 * np.sin(2 * np.pi * (x)) + epsilon

    train_size = 32
    noise = 1.0

    X = np.linspace(-0.5, 0.5, train_size).reshape(-1, 1)
    y = f(X, sigma=noise)
    y_true = f(X, sigma=0.0)

    plt.scatter(X, y, marker='+', label='Training data')
    plt.plot(X, y_true, label='Truth')
    plt.title('Noisy training data and ground truth')
    plt.legend()
    plt.show()

    mlp = BayesMLP(input_dim=X.shape[1],
                   hidden_dims=[20, 20, 1],
                   acts=['relu', 'relu', None],
                   # priors=LaplacePrior(mu=0, b=1.0)
                   # priors=GaussPrior(mu=0, sigma=1.0, backend=K)
                   priors=GaussMixturePrior(mus=[0, 0], sigmas=[1.5, 0.1],
                                            pis=[0.5, 0.5], backend=K),
                   ).model
    mlp.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.08), metrics=['mse'])
    mlp.fit(X, y, batch_size=X.shape[0], epochs=1500)

    X_test = np.linspace(-1.5, 1.5, 1000).reshape(-1, 1)
    y_pred_list = []

    for i in range(500):
        y_pred = mlp.predict(X_test)
        y_pred_list.append(y_pred)

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
