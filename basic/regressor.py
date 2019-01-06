# -*- coding: utf-8 -*-

import numpy as np
from .utils import *


## Ridge regression
# X: N by P feature matrix, N number of samples, P number of features
# y: N by 1 target vector
# b: P by 1 regression coefficients
# b0: the intercept
def ridgereg(y, X, coeff = 1e-4):
    N, P = X.shape
    PHI = np.concatenate((np.ones([N,1]), X), axis=1)
    if (P > N):
        invC = woodburyinv(coeff * np.eye(P+1), PHI.T, PHI, np.eye(N))
    else:
        invC = np.linalg.inv(coeff*np.eye(P+1)+ np.dot(PHI.T, PHI))
    w = np.dot(np.dot(invC, PHI.T), y)
    b = w[1:]
    b0 = w[0]
    return b, b0