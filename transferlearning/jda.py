# -*- coding: utf-8 -*-

# reference: https://github.com/jindongwang/transferlearning/
# Pan S J, Tsang I W, Kwok J T, et al.
# Domain adaptation via transfer component analysis[J].
# IEEE Transactions on Neural Networks, 2011, 22(2): 199-210.

import scipy as sp
import sklearn.metrics
from kernel.kernel import *


class JDA(object):
    def __init__(self, dim=30, beta=1, kernel=None, kargs=None, maxit=10):
        self.kernel = kernel
        self.dim = dim
        self.beta = beta
        self.kargs = kargs # args for the kernel function
        self.maxit = maxit

    def fit(self, Xs, Ys, Xt, Yt):
        Ns, Nt = len(Xs), len(Xt)
        X = np.vstack((Xs, Xt))
        N, P = X.shape
        C = len(np.unique(Ys))
        X /= np.expand_dims(np.linalg.norm(X, axis=1), P)  # normalization
        E = np.vstack((1 / Ns * np.ones((Ns, 1)), -1 / Nt * np.ones((Nt, 1))))
        H = np.eye(N) - 1 / N * np.ones((N, N))

        Yp = None
        acc_list = []
        for i in range(self.maxit):
            Mc = 0
            M0 = E * E.T * C
            if Yp is not None and len(Yp) == Nt:
                for c in range(1, C + 1):
                    E = np.zeros((N, 1))
                    tt = Ys == c
                    E[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                    yy = Yp == c
                    E[np.array(np.where(yy == True)) + Ns] = \
                        -1 / len(Yp[np.where(Yp == c)])
                    E[np.isinf(E)] = 0
                    Mc = Mc + np.dot(E, E.T)
            M = M0 + Mc

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

            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(Xs_new, Ys.ravel())
            Yp = knn.predict(Xt_new)
            acc = sklearn.metrics.accuracy_score(Yt, Yp)
            acc_list.append(acc)
            print('JDA iteration [{}/{}]: Acc: {:.4f}'
                  .format(i + 1, self.maxit, acc))

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
                jda = JDA(dim=30, beta=1, kernel=None, kargs=[1], maxit=10)

                Xs_new, Xt_new = jda.fit(Xs, Ys, Xt, Yt)
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(Xs_new, Ys.ravel())
                Yp = knn.predict(Xt_new)
                acc = sklearn.metrics.accuracy_score(Yt, Yp)
                print(acc)
                # It should print 0.910828025477707