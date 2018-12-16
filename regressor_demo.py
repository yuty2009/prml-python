# -*- coding: utf-8 -*-
## Least Square Regression with norm2 regularization
#
# approximate y = sin(2*pi*[0:0.01:1]) by
# y = w0 + w1*x + w2*x^2 + w3*x^3 + ...+ wn*x^n
# min{ |y - t|^2 + lambda*w'*w }
#

import numpy as np
import matplotlib.pyplot as plt
from basic.regressor import ridgereg

def polybasis(x, order):
    vector = np.zeros([len(x),order+1])
    for i in range(order+1):
        vector[:,i] = np.power(x, i)
    return vector

if __name__ == "__main__":

    sigma = 0.3
    t = np.linspace(0.01, 1, 100)
    N = len(t)
    order = 9

    perm1 = np.random.permutation(N)
    x1 = t[perm1]
    y1 = np.sin(2 * np.pi * x1) + sigma * np.random.randn(N)
    PHI1 = polybasis(x1, order)

    # train the model
    w, b = ridgereg(y1, PHI1, 1e-4)

    # generate testing samples
    perm2 = np.random.permutation(N)
    x2 = t[perm2]
    y2 = np.sin(2 * np.pi * x2) + sigma * np.random.randn(N)
    PHI2 = polybasis(x2, order)

    # predict
    yp = np.matmul(PHI2, w) + b

    tt = np.sin(2 * np.pi * t)
    PHIt = polybasis(t, order)
    tp = np.matmul(PHIt, w) + b

    # visualization
    plt.figure(1)
    plt.plot(t, tt, '-r')
    plt.plot(t, tp, '-b')
    plt.plot(x1, y1, 'ob')
    plt.plot(x2, y2, 'og')
    plt.legend(['standard sin(x)', 'predicted curve', 'trainset', 'testset'])
    plt.show(block=True)
