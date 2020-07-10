# -*- coding: utf-8 -*-

# reference: https://github.com/jindongwang/transferlearning/
# Pan S J, Tsang I W, Kwok J T, et al.
# Domain adaptation via transfer component analysis[J].
# IEEE Transactions on Neural Networks, 2011, 22(2): 199-210.

import scipy as sp
from kernel.kernel import *


class TCA(object):
    def __init__(self, dim=30, beta=1, kernel=None, kargs=None):
        self.kernel = kernel
        self.dim = dim
        self.beta = beta
        self.kargs = kargs # args for the kernel function

    def fit(self, Xs, Xt):
        Ns, Nt = len(Xs), len(Xt)
        X = np.vstack((Xs, Xt))
        N, P = X.shape
        X /= np.expand_dims(np.linalg.norm(X, axis=1), P) # normalization
        E = np.vstack((1 / Ns * np.ones((Ns, 1)), -1 / Nt * np.ones((Nt, 1))))
        M = E * E.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(N) - 1 / N * np.ones((N, N))
        if self.kernel is None:
            N_eye = P
            K = X.T
        else:
            N_eye = N
            K = kernel(np.asarray(X), np.asarray(X), self.kernel, self.kargs)
        A = np.linalg.multi_dot([K, M, K.T]) + self.beta * np.eye(N_eye)
        B = np.linalg.multi_dot([K, H, K.T])
        V, U = sp.linalg.eig(A, B)
        ind = np.argsort(V)
        A = U[:, ind[:self.dim]]
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :Ns].T, Z[:, Ns:].T
        return Xs_new, Xt_new


if __name__ == '__main__':

    import scipy.io as sio
    import sklearn.metrics
    from sklearn.neighbors import KNeighborsClassifier

    datapath = 'e:/prmldata/office_caltech_surf_zscore/'
    domains = ['caltech10_zscore_SURF_L10.mat',
               'amazon_zscore_SURF_L10.mat',
               'webcam_zscore_SURF_L10.mat',
               'dslr_zscore_SURF_L10.mat']

    for i in [2]:
        for j in [3]:
            if i != j:
                src, tar = datapath + domains[i], datapath + domains[j]
                src_domain, tar_domain = sio.loadmat(src), sio.loadmat(tar)
                Xs, Ys = src_domain['Xt'], src_domain['Yt']
                Xt, Yt = tar_domain['Xs'], tar_domain['Ys']
                tca = TCA(dim=30, beta=1, kernel='rbf', kargs=[1])

                Xs_new, Xt_new = tca.fit(Xs, Xt)
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(Xs_new, Ys.ravel())
                Yp = knn.predict(Xt_new)
                acc = sklearn.metrics.accuracy_score(Yt, Yp)
                print(acc)
                # It should print 0.910828025477707