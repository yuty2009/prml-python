# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from common.utils import *
from bayesian.linear import *


N1 = 128 # number of samples in class 1
N2 = 128 # number of samples in class 2
N = N1 + N2
P = 1024 # feature dimension
X1 = np.random.randn(N1,P)
X2 = np.random.randn(N2,P) - 10
XRaw = np.concatenate((X1, X2), axis=0)
# designed weights
NG = 32 # number of groups
PG = P // NG # number of feature per-group
groups = np.arange(P) // PG + 1
NSG = 10 # number of active groups
perm = np.random.permutation(NG)
actives = perm[:NSG] + 1
w0 = np.zeros(P)
for i in range(NSG):
    indices = np.squeeze(np.argwhere(groups == actives[i]))
    # w0[indices] = np.random.randn(len(indices)) # gaussian signal
    w0[indices] = np.ones(len(indices)) # uniform signal
# design matrix and class label
sigma = 0.2
X0 = np.dot(XRaw, np.diag(w0)) + sigma*np.random.randn(N,P)
y0 = np.concatenate((np.ones(N1), -1*np.ones(N2)), axis=0)
# permutate the samples
perm = np.random.permutation(N)
X = X0[perm,:]
y = y0[perm]
# calculate the discriminability of each feature
rr = rsquare(X, y)

bardlog = BayesARDLogisticRegression()
w1, b1 = bardlog.fit(X, y)
bgardlog = BayesGARDLogisticRegression()
w2, b2 = bgardlog.fit(X, y, NG)

# visualize
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(w0)
plt.axis('tight')
plt.ylim([-2, 2])
plt.title('raw')
plt.subplot(3,1,2)
plt.plot(w1)
plt.axis('tight')
# plt.ylim([-2, 2])
plt.title('bardlog')
plt.subplot(3,1,3)
plt.plot(w2)
plt.axis('tight')
# plt.ylim([-2, 2])
plt.title('bgardlog')
plt.show()