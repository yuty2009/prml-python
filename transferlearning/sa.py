# -*- coding: utf-8 -*-

# reference: https://github.com/jindongwang/transferlearning/
# Fernando B, Habrard A, Sebban M, et al. Unsupervised visual
# domain adaptation using subspace alignment[C]//ICCV. 2013: 2960-2967.

import numpy as np
from basic.linear import PCA


class SA(object):
    def __init__(self, dim=30):
        self.dim = dim

    def fit(self, Xs, Xt):
        Us, _ = PCA(Xs)
        Ut, _ = PCA(Xt)
        Us = Us[:, 0:self.dim]
        Ut = Ut[:, 0:self.dim]
        Ua = np.linalg.multi_dot([Us, Us.T, Ut])
        Xs_new = np.dot(Xs, Ua)
        Xt_new = np.dot(Xt, Ut)
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
                SA = SA(dim=100)

                Xs_new, Xt_new = SA.fit(Xs, Xt)
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(Xs_new, Ys.ravel())
                Yp = knn.predict(Xt_new)
                acc = sklearn.metrics.accuracy_score(Yt, Yp)
                print(acc)