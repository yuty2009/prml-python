# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from basic.utils import *
# from bayesian.regressor import *
from bayesian.pymc3 import *


N = 256
P = 1024 # feature dimension
X = np.random.randn(N,P)
# designed weights
NG = 32 # number of groups
PG = np.floor(P/NG).astype(int) # number of feature per-group
groups = np.ceil(np.arange(P)/PG).astype(int)
NSG = 10 # number of active groups
perm = np.random.permutation(NG-1)+1
actives = perm[:NSG]
w0 = np.zeros(P)
for i in range(NSG):
    indices = np.squeeze(np.argwhere(groups == actives[i]))
    w0[indices] = np.random.randn(len(indices)) # gaussian signal
    # w0[indices] = np.ones(len(indices)) # uniform signal
# design matrix and class label
sigma = 0.2
y = np.dot(X, w0) + sigma*np.random.randn(N)

w1, b1 = bardreg(y, X)
w2, b2 = bgardreg(y, X, NG)

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
