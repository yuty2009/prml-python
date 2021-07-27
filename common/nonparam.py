# -*- coding: utf-8 -*-

import math
import numpy as np


def knn(x, y, kn, fdist = 'euclidean'):
    x = np.array(x)
    y = np.array(y)
    if len(x.shape) == 1: x = np.expand_dims(x, -1)
    if len(y.shape) == 1: y = np.expand_dims(y, -1)
    N1, d = x.shape
    N2, d_ = y.shape
    assert d == d_, 'x and y shape mismatch'

    p = []
    for i in range(N2):
        dists = []
        y1 = y[i, :]
        for j in range(N1):
            x1 = x[j, :]
            if fdist == 'euclidean':
                dist = np.linalg.norm(y1 - x1)
            else:
                print('unknown distance function')
            dists.append(dist)
        dists.sort()
        v = (2*dists[kn-1])**d
        p1 = kn / (N1 * v)
        p.append(p1)
    return np.array(p)


def parzon(x, y, h1=1, fwin='gauss'):
    x = np.array(x)
    y = np.array(y)
    if len(x.shape) == 1: x = np.expand_dims(x, -1)
    if len(y.shape) == 1: y = np.expand_dims(y, -1)
    N1, d = x.shape
    N2, d_ = y.shape
    assert d == d_, 'x and y shape mismatch'

    h = h1 / math.sqrt(N1)
    p = []
    for i in range(N2):
        p1 = 0
        y1 = y[i, :]
        for j in range(N1):
            x1 = x[j, :]
            u = (y1 - x1) / h
            if fwin == 'gauss':
                p1 = p1 + gauss(u)
            elif fwin == 'cube':
                p1 = p1 + cube(u)
            else:
                print('unknown window type')
        p1 = p1 / (N1 * h**d)
        p.append(p1)
    return np.array(p)


def gauss(u):
    u = np.array(u)
    return math.exp(-0.5*np.dot(u, u))/math.sqrt(2*math.pi)


def cube(u):
    T = abs(u)
    if all(t <= 0.5 for t in T):
        return 1
    else:
        return 0


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    '''
    x0 = np.random.randn(100000, 1)
    y = np.arange(-3, 3, 0.01)
    Ns = [10, 100, 1000]
    hs = [0.25, 1, 4]
    plt.figure(1)
    for i, N in enumerate(Ns):
        x = x0[:N, :]
        for j, h in enumerate(hs):
            p = parzon(x, y, h)
            plt.subplot(len(Ns), len(hs), i*len(hs)+j+1)
            plt.plot(y, p, 'k-')
            plt.title("n = %d, h = %.2f" % (N, h))
    plt.show()
    '''


    x0 = np.random.multivariate_normal([0, 0], np.eye(2), 100000)
    y1, y2 = np.meshgrid(np.arange(-3, 3, 0.25), np.arange(-3, 3, 0.25))
    y = np.array([y1.flatten(), y2.flatten()]).transpose()
    Ns = [100, 500, 2000]
    hs = [0.25, 1, 4]
    ks = [1, 10, 50]
    plt.figure(1)
    for i, N in enumerate(Ns):
        x = x0[:N, :]
        # for j, kn in enumerate(ks):
        for j, h in enumerate(hs):
            # p = knn(x, y, kn)
            p = parzon(x, y, h)
            ax = plt.subplot(len(Ns), len(hs), i*len(hs)+j+1, projection='3d')
            ax.plot_surface(y1, y2, p.reshape(y1.shape))
            # plt.title("n = %d, kn = %d" % (N, kn))
            plt.title("n = %d, h = %.2f" % (N, h))
    plt.show()

