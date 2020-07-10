# -*- coding: utf-8 -*-

# reference: https://github.com/jindongwang/transferlearning/
# Sun B, Feng J, Saenko K. Return of frustratingly easy domain
# adaptation[C]//AAAI. 2016, 6(7): 8.

import numpy as np
import scipy as sp


class CORAL(object):
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(sp.linalg.fractional_matrix_power(cov_src, -0.5),
                         sp.linalg.fractional_matrix_power(cov_tar, 0.5))
        A_coral = np.real(A_coral) # retain real part only
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new


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
                coral = CORAL()

                Xs_new = coral.fit(Xs, Xt)
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(Xs_new, Ys.ravel())
                Yp = knn.predict(Xt)
                acc = sklearn.metrics.accuracy_score(Yt, Yp)
                print(acc)