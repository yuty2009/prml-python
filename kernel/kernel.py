# -*- coding: utf-8 -*-

import numpy as np


class RBFKernel(object):
    def __init__(self, gamma=None):
        self.gamma = gamma

    def eval(self, X, Y):
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]

        N1 = X.shape[0]
        N2 = Y.shape[0]
        if X.shape[1] != Y.shape[1]:
            assert 'Kernel of two variables with different dim'

        K = np.zeros((N1, N2), dtype=X.dtype)
        for i in range(N1):
            xi = X[i]
            for j in range(N2):
                xj = Y[j]
                K[i, j] = np.exp(-np.dot(xi - xj, xi - xj) / (2 * self.gamma ** 2))
        return K

    def deriv(self, X, Y):
        N1 = X.shape[0]
        N2 = Y.shape[0]
        if X.shape[1] != Y.shape[1]:
            assert 'Kernel of two variables with different dim'

        dK = []
        for i in range(N1):
            xi = X[i]
            for j in range(N2):
                xj = Y[j]
                dK1 = np.zeros((N1, N2), dtype=X.dtype)
                dK1[i, j] = (self.gamma ** -3) * np.dot(xi - xj, xi - xj) \
                            * np.exp(-np.dot(xi - xj, xi - xj)
                                     / (2 * self.gamma ** 2))
                dK = [dK1]
        return dK


def kernel(X, Y, type='linear', args=None):
    """
    Evaluates kernel function.
    X : array of shape (n_samples_1, n_features)
    Y : array of shape (n_samples_2, n_features)
    where k: a x b -> R is a kernel function given by identifier type
    and argument arg:
    Identifier    Name           Definition
    'linear'  ... linear kernel  k(a,b) = a'*b
    'poly'    ... polynomial     k(a,b) = (a'*b+arg[2])^arg[1]
    'rbf'     ... RBF (Gaussian) k(a,b) = exp(-0.5*||a-b||^2/arg[1]^2)
    'sigmoid' ... Sigmoidal      k(a,b) = tanh(arg[1]*(a'*b)+arg[2])
    """
    N1 = X.shape[0]
    N2 = Y.shape[0]
    if X.shape[1] != Y.shape[1]:
        assert 'Kernel of two variables with different dim'

    K = np.zeros((N1, N2), dtype=X.dtype)
    for i in range(N1):
        xi = X[i]
        for j in range(N2):
            xj = Y[j]
            if isinstance(type, str) and type.lower() == 'linear':
                K[i, j] = np.dot(xi, xj)
            elif isinstance(type, str) and type.lower() == 'poly':
                K[i, j] = (args[0] * np.dot(xi, xj) + args[1]) ** args[2]
            elif isinstance(type, str) and (type.lower() == 'gaussian' or
                                            type.lower() == 'rbf'):
                K[i, j] = np.exp(-np.dot(xi - xj, xi - xj) / (2 * args[0] * args[0]))
            elif isinstance(type, str) and (type.lower() == 'sigmoid' or
                                            type.lower() == 'tanh'):
                K[i, j] = np.tanh(args[0] * np.dot(xi, xj) + args[1])
            elif isinstance(type, str) and type.lower() == 'gpkernel':
                K[i, j] = args[0] * np.exp(-(args[1] / 2) * np.dot(xi - xj, xi - xj)) \
                          + args[2] + args[3] * np.dot(xi, xj)
            else:
                assert 'Unknown kernel type'
    return K


def kderiv(X, Y, type, args):
    """
    Evaluates derivatives of kernel function wrt. the args.
    X : array of shape (n_samples_1, n_features)
    Y : array of shape (n_samples_2, n_features)
    where k: a x b -> R is a kernel function given by identifier type
    and argument arg:
    Identifier    Name           Definition
    'linear'  ... linear kernel  k(a,b) = a'*b
    'poly'    ... polynomial     k(a,b) = (a'*b+arg[2])^arg[1]
    'rbf'     ... RBF (Gaussian) k(a,b) = exp(-0.5*||a-b||^2/arg[1]^2)
    'sigmoid' ... Sigmoidal      k(a,b) = tanh(arg[1]*(a'*b)+arg[2])
    """
    N1 = X.shape[0]
    N2 = Y.shape[0]
    if X.shape[1] != Y.shape[1]:
        assert 'Kernel of two variables with different dim'

    dK = []
    for i in range(N1):
        xi = X[i]
        for j in range(N2):
            xj = Y[j]
            if isinstance(type, str) and type.lower() == 'poly':
                dK1 = np.zeros((N1, N2), dtype=X.dtype)
                dK2 = np.zeros((N1, N2), dtype=X.dtype)
                dK3 = np.zeros((N1, N2), dtype=X.dtype)
                dK1[i, j] = np.dot(xi, xj) \
                            * args[2] * (args[0]*np.dot(xi, xj) + args[1]) ** (args[2] - 1)
                dK2[i, j] = args[2] * (args[0]*np.dot(xi, xj) + args[1]) ** (args[2] - 1)
                dK3[i, j] = np.log(args[0]*np.dot(xi, xj) + args[1]) \
                            * (args[0]*np.dot(xi, xj) + args[1]) ** args[2]
                dK = [dK1, dK2, dK3]
            elif isinstance(type, str) and (type.lower() == 'gaussian' or
                                            type.lower() == 'rbf'):
                dK1 = np.zeros((N1, N2), dtype=X.dtype)
                dK1[i, j] = (args[0] ** -3) * np.dot(xi - xj, xi - xj) \
                            * np.exp(-np.dot(xi - xj, xi - xj)
                                     / (2 * args[0] * args[0]))
                dK = [dK1]
            elif isinstance(type, str) and (type.lower() == 'sigmoid' or
                                            type.lower() == 'tanh'):
                dK1 = np.zeros((N1, N2), dtype=X.dtype)
                dK2 = np.zeros((N1, N2), dtype=X.dtype)
                dK1[i, j] = (1 - np.tanh(args[0] * np.dot(xi, xj) + args[1]) ** 2)\
                            * np.dot(xi, xj)
                dK2[i, j] = 1 - np.tanh(args[0] * np.dot(xi, xj) + args[1]) ** 2
                dK = [dK1, dK2]
            elif isinstance(type, str) and type.lower() == 'gpkernel':
                dK1 = np.zeros((N1, N2), dtype=X.dtype)
                dK2 = np.zeros((N1, N2), dtype=X.dtype)
                dK3 = np.zeros((N1, N2), dtype=X.dtype)
                dK4 = np.zeros((N1, N2), dtype=X.dtype)
                dK1[i, j] = np.exp(-(args[1] / 2) * np.dot(xi - xj, xi - xj))
                dK2[i, j] = -args[0] * 0.5 * np.dot(xi - xj, xi - xj) \
                            * np.exp(-(args[1] / 2) * np.dot(xi - xj, xi - xj))
                dK3[i, j] = 1
                dK4[i, j] = np.dot(xi, xj)
                dK = [dK1, dK2, dK3, dK4]
            else:
                assert 'Unknown kernel type'
    return dK


if __name__ == "__main__":

    a = np.random.randn(4, 3)
    b = np.random.randn(5, 3)

    c = kernel(a, b)
    print(c)