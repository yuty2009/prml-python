# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from common.utils import *
# from bayesian.linear import *
from bayesian.pytorch import *
# from bayesian.pymc3 import *


N = 256
P = 1024 # feature dimension
X = np.random.randn(N,P)
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
    w0[indices] = np.random.randn(len(indices)) # gaussian signal
    # w0[indices] = np.ones(len(indices)) # uniform signal
# design matrix and class label
sigma = 0.2
y = np.dot(X, w0) + sigma*np.random.randn(N)

bayesreg = BayesLinearRegression()
w1, b1 = bayesreg.fit(X, y)
bgardreg = BayesGARDLinearRegression()
w2, b2 = bgardreg.fit(X, y, NG)

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
plt.ylim([-2, 2])
plt.title('bardreg')
plt.subplot(3,1,3)
plt.plot(w2)
plt.axis('tight')
plt.ylim([-2, 2])
plt.title('bgardreg')
plt.show()
