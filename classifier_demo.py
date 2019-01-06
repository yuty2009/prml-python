# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from basic.classifier import *
from bayesian.classifier import *


if __name__ == "__main__":

    M = 2
    # generate train dataset
    N1 = 20 # number of samples in class 1
    N2 = 30 # number of samples in class 2
    X1 = np.random.randn(N1,M)
    X2 = np.random.randn(N2,M) + np.concatenate((2*np.ones([N2,1]), np.ones([N2,1])), axis=1)

    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((np.ones(N1), -1*np.ones(N2)))

    # train the model
    W_lda, b_lda = FLDA(y, X)
    # W_log, b_log = logistic(y, X)
    # W_log, b_log = bayeslog(y, X)
    W_log, b_log = bardlog(y, X)

    y1 = np.sign(np.matmul(X1, W_lda) + b_lda)
    y2 = np.sign(np.matmul(X2, W_lda) + b_lda)

    t = np.linspace(min(X[:,1]),max(X[:,1]), 100)
    v1 = (-W_lda[0]*t - b_lda) / W_lda[1]
    v2 = (-W_log[0]*t - b_log) / W_log[1]

    plt.figure(1)
    plt.plot(t, v1, 'g-')
    plt.plot(t, v2, 'b-')
    plt.scatter(X1[:,0], X1[:,1], marker='x', c='b')
    plt.scatter(X2[:,0], X2[:,1], marker='o', c='r')
    plt.legend(['LDA', 'Logistic', 'c1', 'c2'])
    plt.show(block=True)
