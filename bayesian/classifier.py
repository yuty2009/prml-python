# -*- coding: utf-8 -*-

import copy
import numpy as np
from basic.utils import *


## Bayesian logistic regression
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector of labels
# b: P by 1 regression coefficients
# b0: the intercept
def bayeslog(y, X):
    N, P = X.shape
    PHI = np.concatenate((np.ones([N, 1]), X), axis=1)
    y[np.argwhere(y == -1)] = 0 # the class label should be[1 0]

    # initialize with the least square estimation
    coeff = 1e-4
    if (P > N):
        invC = woodburyinv(coeff * np.eye(P + 1), PHI.T, PHI, np.eye(N))
    else:
        invC = np.linalg.inv(coeff * np.eye(P + 1) + np.dot(PHI.T, PHI))
    w = np.dot(np.dot(invC, PHI.T), y)
    # w = ones(P, 1) # rough initialization

    # hyperparameter
    alpha = 1

    # stop conditions
    d_w = np.Inf
    maxit = 500
    stopeps = 1e-6

    i = 0
    while (d_w > stopeps) and (i < maxit):
        wold = copy.deepcopy(w)

        ## E step(IRLS update)
        t = 1/(1 + np.exp(-np.dot(PHI, w))) # predicted target value
        diagR = t * (1 - t)
        R = np.diag(diagR) # the variance matrix of target value
        invR = np.diag(1/diagR)
        if (P > N):
            Sigma = woodburyinv(alpha*np.eye(P+1), PHI.T, PHI, invR)
        else:
            Sigma = np.linalg.inv(alpha*np.eye(P+1) + np.dot(np.dot(PHI.T, R), PHI))
        w = w - np.dot(Sigma, np.dot(PHI.T, t-y)+alpha*w)

        # M step
        # [v, d] = np.linalg.eig(np.dot(np.dot(PHI.T,R),PHI))
        d = myeig(np.dot(np.diag(np.sqrt(diagR)), PHI))
        gamma = sum(d / (alpha + d))
        alpha = gamma / np.dot(w.T, w)

        evidence = (P/2) * np.log(alpha) + sum(y*np.log(t)+(1-y)*np.log(1-t)) \
                   - (alpha/2)*np.dot(w.T, w) - 0.5*np.sum(np.log((d+alpha)))

        d_w = np.linalg.norm(w - wold) / (np.linalg.norm(wold) + 1e-32)

        print('Iteration %i: evidence = %f, wchange = %f, alpha = %f'
              % (i, evidence, d_w, alpha))
        i += 1

    if i < maxit:
        print('Optimization of alpha and beta successful.')
    else:
        print('Optimization terminated due to max iteration.')

    b = w[1:]
    b0 = w[0]
    return b, b0


## Bayesian logistic regression with ARD prior
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector of labels
# b: P by 1 regression coefficients
# b0: the intercept
def bardlog(y, X):
    N, P = X.shape
    PHI = np.concatenate((np.ones([N, 1]), X), axis=1)
    y[np.argwhere(y == -1)] = 0 # the class label should be[1 0]

    # initialize with the least square estimation
    coeff = 1e-4
    if (P > N):
        invC = woodburyinv(coeff * np.eye(P + 1), PHI.T, PHI, np.eye(N))
    else:
        invC = np.linalg.inv(coeff * np.eye(P + 1) + np.dot(PHI.T, PHI))
    w = np.dot(np.dot(invC, PHI.T), y)
    # w = np.ones(P+1) # rough initialization

    # hyperparameter
    alphas = np.ones(P+1)  # difference with bayeslog

    # stop conditions
    d_w = np.Inf
    maxit = 500
    stopeps = 1e-6
    maxalpha = 1e9
    eps = 2**(-52)

    i = 0
    while (d_w > stopeps) and (i < maxit):
        wold = copy.deepcopy(w)

        # eliminate very large alphas to avoid precision problem of sigma
        index0 = np.argwhere(alphas > min(alphas) * maxalpha)
        index1 = np.setdiff1d(np.arange(P + 1), index0)
        if len(index1) <= 0:
            print('Optimization terminated due that all alphas are large.')
            break
        alphas1 = alphas[index1]
        PHI1 = PHI[:, index1]
        w1 = w[index1]

        ## E step(IRLS update)
        t = 1/(1 + np.exp(-np.dot(PHI, w))) # predicted target value
        diagR = t * (1 - t)
        R = np.diag(diagR) # the variance matrix of target value
        invR = np.diag(1/diagR)
        N1, P1 = PHI1.shape
        if (P1 > N1):
            Sigma1 = woodburyinv(np.diag(alphas1), PHI1.T, PHI1, invR)
        else:
            Sigma1 = np.linalg.inv(np.diag(alphas1) + np.dot(np.dot(PHI1.T, R), PHI1))
        w1 = w1 - np.dot(Sigma1, np.dot(PHI1.T, t-y) + np.dot(np.diag(alphas1), w1))
        w[index1] = w1
        if len(index0) > 0: w[index0] = 0

        # M step
        # [v, d] = np.linalg.eig(np.dot(np.dot(PHI.T,R),PHI))
        d = myeig(np.dot(np.diag(np.sqrt(diagR)), PHI))
        gamma1 = 1 - alphas1 * np.diag(Sigma1)
        alphas1 = np.maximum(gamma1, eps) / (np.dot(w1.T, w1) + 1e-32)
        alphas[index1] = alphas1

        evidence = 0.5*np.sum(np.log(alphas)) + sum(y*np.log(t)+(1-y)*np.log(1-t)) \
                   - 0.5*np.dot(np.dot(w.T, np.diag(alphas)), w) \
                   - 0.5*np.sum(np.log((d+alphas)))

        d_w = np.linalg.norm(w - wold) / (np.linalg.norm(wold) + 1e-32)

        print('Iteration %i: evidence = %f, wchange = %f, min(alphas) = %f'
              % (i, evidence, d_w, min(alphas)))
        i += 1

    if i < maxit:
        print('Optimization of alpha and beta successful.')
    else:
        print('Optimization terminated due to max iteration.')

    b = w[1:]
    b0 = w[0]
    return b, b0


## Bayesian logistic regression with grouped ARD prior
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# group: No. of groups or a group id vector
#        e.g. group = [1 1 1 2 2 2 3 3 3 4 4 4]' indicates that there are
#        4 group with 3 members in each
# b: P by 1 regression coefficients
# b0: the intercept
def bgardlog(y, X, group):
    N, P = X.shape
    PHI = np.concatenate((np.ones([N, 1]), X), axis=1)
    y[np.argwhere(y == -1)] = 0  # the class label should be[1 0]

    if np.size(group) == 1:
        group = np.ceil((np.arange(P)+1)/group).astype(int)
    group = np.append([0], group) # account for bias
    groupid = np.unique(group)
    NG = len(groupid)

    w = np.zeros(P+1) # rough initialization
    alphas = np.ones(P+1) # difference with bayesreg
    beta = 10

    # stop conditions
    d_w = np.Inf
    maxit = 500
    stopeps = 1e-4
    maxalpha = 1e9
    eps = 2**(-52)

    i = 0
    while (d_w > stopeps) and (i < maxit):
        wold = copy.deepcopy(w)

        # eliminate very large alphas to avoid precision problem of sigma
        index0 = np.argwhere(alphas > min(alphas) * maxalpha)
        index1 = np.setdiff1d(np.arange(P + 1), index0)
        if len(index1) <= 0:
            print('Optimization terminated due that all alphas are large.')
            break
        alphas1 = alphas[index1]
        PHI1 = PHI[:, index1]
        w1 = w[index1]

        ## E step(IRLS update)
        t = 1 / (1 + np.exp(-np.dot(PHI, w)))  # predicted target value
        diagR = t * (1 - t)
        R = np.diag(diagR)  # the variance matrix of target value
        invR = np.diag(1/diagR)
        N1, P1 = PHI1.shape
        if (P1 > N1):
            Sigma1 = woodburyinv(np.diag(alphas1), PHI1.T, PHI1, invR)
        else:
            Sigma1 = np.linalg.inv(np.diag(alphas1) + np.dot(np.dot(PHI1.T, R), PHI1))
        # one iteration of Newton's gradient descent (w = w - inv(Hess)*Grad)
        # perform this update only once in each E step
        w1 = w1 - np.dot(Sigma1, np.dot(PHI1.T, t - y) + np.dot(np.diag(alphas1), w1))
        w[index1] = w1
        if len(index0) > 0: w[index0] = 0

        # M step
        # [v, d] = np.linalg.eig(np.dot(np.dot(PHI.T,R),PHI))
        d = myeig(np.dot(np.diag(np.sqrt(diagR)), PHI))

        gamma1 = 1 - alphas1 * np.diag(Sigma1)
        gamma = np.zeros(alphas.shape)
        gamma[index1] = gamma1

        for g in range(NG):
            index_ig = np.argwhere(group == groupid[g])
            w_ig = w[index_ig]
            if np.linalg.norm(w_ig) == 0: continue
            gamma_ig = gamma[index_ig]
            alpha_ig = sum(gamma_ig) / np.dot(w_ig.T,w_ig)
            alphas[index_ig] = alpha_ig

        evidence = 0.5 * np.sum(np.log(alphas)) + sum(y * np.log(t) + (1 - y) * np.log(1 - t)) \
                   - 0.5 * np.dot(np.dot(w.T, np.diag(alphas)), w) \
                   - 0.5 * np.sum(np.log((d + alphas)))

        d_w = np.linalg.norm(w - wold) / (np.linalg.norm(wold) + 1e-32)

        print('Iteration %i: evidence = %f, wchange = %f, min(alphas) = %f'
              % (i, evidence, d_w, min(alphas)))
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
