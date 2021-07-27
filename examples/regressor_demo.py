# -*- coding: utf-8 -*-
#
# Least Square Regression with norm2 regularization
#
# approximate y = sin(2*pi*[0:0.01:1]) by
# y = w0 + w1*x + w2*x^2 + w3*x^3 + ...+ wn*x^n
# min{ |y - t|^2 + lambda*w'*w }

from common.linear import *
from bayesian.linear import *
# from bayesian.pymc3 import *
# from bayesian.pytorch import *


def polybasis(x, order):
    vector = np.zeros([len(x),order+1]).astype(np.float32)
    for i in range(order+1):
        vector[:,i] = np.power(x, i)
    return vector


def f(x, sigma):
    epsilon = np.random.randn(*x.shape) * sigma
    return np.sin(2 * np.pi * (x)) + epsilon


if __name__ == "__main__":

    sigma = 0.1
    t = np.linspace(-0.5, 0.5, 100)
    N = len(t)
    order = 9

    perm1 = np.random.permutation(N)
    x1 = t[perm1]
    y1 = f(x1, sigma)
    PHI1 = polybasis(x1, order)

    # train the model
    # ridgereg = RidgeRegression()
    # w, b = ridgereg.fit(PHI1, y1)
    # bayesreg = BayesLinearRegression()
    # w, b = bayesreg.fit(PHI1, y1)
    bardreg = BayesARDLinearRegression()
    w, b = bardreg.fit(PHI1, y1)

    # generate testing samples
    perm2 = np.random.permutation(N)
    x2 = t[perm2]
    y2 = f(x2, sigma)
    PHI2 = polybasis(x2, order)

    # predict
    yp = np.matmul(PHI2, w) + b

    tt = f(t, 0)
    PHIt = polybasis(t, order)
    tp = np.matmul(PHIt, w) + b

    # visualization
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(t, tt, '-r')
    plt.plot(t, tp, '-b')
    plt.plot(x1, y1, 'ob')
    plt.plot(x2, y2, 'og')
    plt.legend(['standard sin(x)', 'predicted curve', 'trainset', 'testset'])
    plt.subplot(2, 1, 2)
    plt.plot(w)
    plt.axis('tight')
    plt.show(block=True)
