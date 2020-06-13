# -*- coding: utf-8 -*-

import copy
import numpy as np
from basic.utils import *


## Bayesian linear regression
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# b: P by 1 regression coefficients
# b0: the intercept
def bayesreg(y, X):
    N, P = X.shape
    PHI = np.concatenate((np.ones([N, 1]), X), axis=1)

    # initialize with the least square estimation
    coeff = 1e-4
    if (P > N):
        invC = woodburyinv(coeff * np.eye(P + 1), PHI.T, PHI, np.eye(N))
    else:
        invC = np.linalg.inv(coeff * np.eye(P + 1) + np.dot(PHI.T, PHI))
    w = np.dot(np.dot(invC, PHI.T), y)
    # w = ones(P, 1) # rough initialization

    alpha = 2
    beta = 10

    # stop conditions
    d_w = np.Inf
    maxit = 500
    stopeps = 1e-6
    eps = 2**(-52)

    # [v, d] = np.linalg.eig(np.dot(PHI.T, PHI))
    d = myeig(PHI)

    i = 0
    while (d_w > stopeps) and (i < maxit):
        wold = copy.deepcopy(w)

        if (P > N):
            Sigma = woodburyinv(alpha*np.eye(P+1), PHI.T, PHI, (1/beta)*np.eye(N))
        else:
            Sigma = np.linalg.inv(alpha*np.eye(P+1) + beta*np.dot(PHI.T, PHI))

        w = beta * np.dot(np.dot(Sigma, PHI.T), y)
        gamma = np.sum(beta * d / (alpha + beta * d))
        # Note that np.dot() is the inner product for vectors in Python
        alpha = gamma / np.dot(w.T, w)
        rmse = np.sum((y-np.dot(PHI, w))**2)
        beta = max(N - gamma, eps) / (rmse + 1e-32)

        evidence = (P/2) * np.log(alpha) + (N/2)*np.log(beta) - (beta/2)*rmse \
                   - (alpha/2)*np.dot(w.T, w) - 0.5*np.sum(np.log((beta*d+alpha))) - (N/2)*np.log(2*np.pi)

        d_w = np.linalg.norm(w - wold) / (np.linalg.norm(wold) + 1e-32)

        print('Iteration %i: evidence = %f, wchange = %f, rmse = %f, alpha = %f, beta = %f'
              % (i, evidence, d_w, rmse, alpha, beta))
        i += 1

    if i < maxit:
        print('Optimization of alpha and beta successful.')
    else:
        print('Optimization terminated due to max iteration.')

    b = w[1:]
    b0 = w[0]
    return b, b0


## Bayesian linear regression with ARD prior
# refer to Page 347-348 of PRML book
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# b: P by 1 regression coefficients
# b0: the intercept
def bardreg(y, X):
    N, P = X.shape
    PHI = np.concatenate((np.ones([N, 1]), X), axis=1)

    # initialize with the least square estimation
    coeff = 1e-4
    if (P > N):
        invC = woodburyinv(coeff * np.eye(P + 1), PHI.T, PHI, np.eye(N))
    else:
        invC = np.linalg.inv(coeff * np.eye(P + 1) + np.dot(PHI.T, PHI))
    w = np.dot(np.dot(invC, PHI.T), y)
    # w = ones(P, 1) # rough initialization

    alphas = 2 * np.ones(P + 1)  # difference with bayesreg
    beta = 10

    # stop conditions
    d_w = np.Inf
    maxit = 500
    stopeps = 1e-6
    maxalpha = 1e9
    eps = 2**(-52)

    # [v, d] = np.linalg.eig(np.dot(PHI.T, PHI))
    d = myeig(PHI)

    i = 0
    while (d_w > stopeps) and (i < maxit):
        wold = copy.deepcopy(w)

        # eliminate very large alphas to avoid precision problem of sigma
        index0 = np.argwhere(alphas > min(alphas) * maxalpha)
        index1 = np.setdiff1d(np.arange(P+1), index0)
        if len(index1) <= 0:
            print('Optimization terminated due that all alphas are large.')
            break
        alphas1 = alphas[index1]
        PHI1 = PHI[:, index1]

        if (P > N):
            Sigma1 = woodburyinv(np.diag(alphas1), PHI1.T, PHI1, (1/beta)*np.eye(N))
        else:
            Sigma1 = np.linalg.inv(np.diag(alphas1) + beta*np.dot(PHI1.T, PHI1))

        diagSigma1 = np.diag(Sigma1)
        w1 = beta * np.dot(np.dot(Sigma1, PHI1.T), y)
        w[index1] = w1
        if len(index0) > 0: w[index0] = 0

        gamma1 = 1 - alphas1 * diagSigma1
        # Note that * is dot product for vectors in Python
        alphas1 = np.maximum(gamma1, eps) / (w1.T*w1 + 1e-32)
        alphas[index1] = alphas1
        rmse = np.sum((y-np.dot(PHI, w))**2)
        beta = max(N - np.sum(gamma1), eps) / (rmse + 1e-32)

        evidence = 0.5*np.sum(np.log(alphas)) + (N/2)*np.log(beta) - (beta/2)*rmse \
                   - 0.5*np.dot(np.dot(w.T, np.diag(alphas)), w) \
                   - 0.5*np.sum(np.log((beta*d+alphas))) - (N/2)*np.log(2*np.pi)

        d_w = np.linalg.norm(w - wold) / (np.linalg.norm(wold) + 1e-32)

        print('Iteration %i: evidence = %f, wchange = %f, rmse = %f, min(alphas) = %f, beta = %f'
              % (i, evidence, d_w, rmse, min(alphas), beta))
        i += 1

    if i < maxit:
        print('Optimization of alpha and beta successful.')
    else:
        print('Optimization terminated due to max iteration.')

    b = w[1:]
    b0 = w[0]
    return b, b0


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
    PHI = np.concatenate((np.ones([N, 1]), X), axis=1)

    if np.size(group) == 1:
        PG = np.floor(P/group).astype(int)  # number of feature per-group
        group = np.floor((np.arange(P))/PG).astype(int) + 1
    group = np.append([0], group) # account for bias
    groupid = np.unique(group)
    NG = len(groupid)

    # initialize with the least square estimation
    coeff = 1e-4
    if (P > N):
        invC = woodburyinv(coeff * np.eye(P + 1), PHI.T, PHI, np.eye(N))
    else:
        invC = np.linalg.inv(coeff * np.eye(P + 1) + np.dot(PHI.T, PHI))
    w = np.dot(np.dot(invC, PHI.T), y)
    # w = ones(P, 1) # rough initialization

    alphas = 2 * np.ones(P+1) # difference with bayesreg
    beta = 10

    # stop conditions
    d_w = np.Inf
    maxit = 500
    stopeps = 1e-6
    maxalpha = 1e9
    eps = 2**(-52)

    # [v, d] = np.linalg.eig(np.dot(PHI.T, PHI))
    d = myeig(PHI)

    i = 0
    while (d_w > stopeps) and (i < maxit):
        wold = copy.deepcopy(w)

        # eliminate very large alphas to avoid precision problem of sigma
        index0 = np.argwhere(alphas > min(alphas) * maxalpha)
        index1 = np.setdiff1d(np.arange(P+1), index0)
        if len(index1) <= 0:
            print('Optimization terminated due that all alphas are large.')
            break
        alphas1 = alphas[index1]
        PHI1 = PHI[:, index1]

        if (P > N):
            Sigma1 = woodburyinv(np.diag(alphas1), PHI1.T, PHI1, (1/beta)*np.eye(N))
        else:
            Sigma1 = np.linalg.inv(np.diag(alphas1) + beta*np.dot(PHI1.T, PHI1))

        diagSigma1 = np.diag(Sigma1)
        w1 = beta * np.dot(np.dot(Sigma1, PHI1.T), y)
        w[index1] = w1
        if len(index0) > 0: w[index0] = 0

        gamma1 = 1 - alphas1 * diagSigma1
        gamma = np.zeros(alphas.shape)
        gamma[index1] = gamma1

        for g in range(NG):
            index_ig = np.argwhere(group == groupid[g])
            w_ig = w[index_ig]
            if np.linalg.norm(w_ig) == 0: continue
            gamma_ig = gamma[index_ig]
            # Note that * is dot product for vectors in Python.
            # But w_ig is a matrix here due to w_ig = w[index_ig]
            alpha_ig = np.maximum(sum(gamma_ig), eps) / (np.dot(w_ig.T,w_ig) + 1e-32)
            alphas[index_ig] = alpha_ig

        rmse = np.sum((y-np.dot(PHI, w))**2)
        beta = max(N - np.sum(gamma1), eps) / (rmse + 1e-32)

        evidence = 0.5*np.sum(np.log(alphas)) + (N/2)*np.log(beta) - (beta/2)*rmse \
                   - 0.5*np.dot(np.dot(w.T, np.diag(alphas)), w) \
                   - 0.5*np.sum(np.log((beta*d+alphas))) - (N/2)*np.log(2*np.pi)

        d_w = np.linalg.norm(w - wold) / (np.linalg.norm(wold) + 1e-32)

        print('Iteration %i: evidence = %f, wchange = %f, rmse = %f, min(alphas) = %f, beta = %f'
              % (i, evidence, d_w, rmse, min(alphas), beta))
        i += 1

    if i < maxit:
        print('Optimization of alpha and beta successful.')
    else:
        print('Optimization terminated due to max iteration.')

    b = w[1:]
    b0 = w[0]
    return b, b0


if __name__ == "__main__":
    pass
