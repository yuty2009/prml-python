# -*- coding: utf-8 -*-

import numpy as np


def nmodeprod(X, M, n):
    """
    Calculates the n-Mode Product of a Tensor X with a Matrix M
    B = nmodeprod(X, M, n)
    B = X (x)_n M .. According to the Definition in De Lathauwer (2000)
    with:
    :param X:    (I_1 x I_2 x .. I_n x .. I_N) .. ->  n is in [1..N]
    :param M:    (J   x I_n)
    :param B:    (I_1 x I_2 x .. J x   .. I_N)
    note: "(x)_n" is the operator between the tensor and the matrix
    """
    ashape = X.shape
    order = len(ashape)
    mshape = M.shape
    assert n <= order-1 and n >= 0, \
        "Mode n = %d should be no larger than order(X) = %d" % (n, order)
    assert ashape[n] == mshape[1], \
        "The nth dimension of X should be equal to the number of columns " \
        "in M while %d vs %d" % (ashape[n], mshape[1])

    if (n == 0):
        newshape = np.arange(order)
        oldshape = newshape
    elif (n == order-1):
        newshape = [n] + list(np.arange(n))
        oldshape = list(1+np.arange(n)) + [0]
    else:
        newshape = [n] + list(np.arange(n)) + list(np.arange(n+1, order))
        oldshape = list(1+np.arange(n)) + [0] + list(np.arange(n+1, order))

    Xn = np.transpose(X, newshape)
    B = np.dot(M, Xn)
    B = np.transpose(B, oldshape)

    return B


if __name__ == "__main__":

    from PIL import Image
    import matplotlib.pyplot as plt

    imfile = 'basic/rice.png'
    im = Image.open(imfile)
    im = np.asarray(im)
    im = im.astype(np.float32) / 255.0
    sx, sy = im.shape

    gradmatx = np.zeros((sx, sx-1))
    gradmatx[:sx-1, :sx-1] = gradmatx[:sx-1, :sx-1] + np.eye(sx-1)
    gradmatx[1:sx , :sx-1] = gradmatx[1:sx , :sx-1] - np.eye(sx-1)
    gradmaty = np.zeros((sy-1, sy))
    gradmaty[:sy-1, :sy-1] = gradmaty[:sy-1, :sy-1] + np.eye(sy-1)
    gradmaty[:sy-1, 1:sy ] = gradmaty[:sy-1, 1:sy ] - np.eye(sy-1)

    gradmat = 2*gradmaty
    imgradx = nmodeprod(im, gradmat, 1)
    imgrady = nmodeprod(im, gradmat, 0)
    imgradxy = nmodeprod(nmodeprod(im, gradmat, 1), gradmat, 0)

    plt.figure()
    ax = plt.subplot(221); ax.imshow(im)
    ax = plt.subplot(222); ax.imshow(imgradx)
    ax = plt.subplot(223); ax.imshow(imgrady)
    ax = plt.subplot(224); ax.imshow(imgradxy)
    plt.show()