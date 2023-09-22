# -*- coding: utf-8 -*-

import numpy as np
from abc import ABCMeta, abstractmethod

class AbstractActivation(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def compute(self, x):
        raise NotImplementedError()

    @abstractmethod
    def deriv(self, y):
        raise NotImplementedError()


class Linear(AbstractActivation):
    def compute(self, x):
        return x

    def deriv(self, y):
        return 1.

class Sigmoid(AbstractActivation):
    def compute(self, x):
        return 1. / (1. + np.exp(-x))

    def deriv(self, y):
        return y * (1. - y)

class Tanh(AbstractActivation):
    def compute(self, x):
        return np.tanh(x)

    def deriv(self, y):
        return 1. - y**2

class Relu(AbstractActivation):
    def compute(self, x):
        return np.maximum(0, x)

    def deriv(self, y):
        return 1. * (y > 0)

class LeakyRelu(AbstractActivation):
    def compute(self, x):
        return np.maximum(0.01, x)

    def deriv(self, y):
        g = 1. * (y > 0)
        g[g == 0.] = 0.01
        return g

class Softmax(AbstractActivation):
    def compute(self, X):
        expvx = np.exp(X - np.max(X, axis=-1)[..., np.newaxis])
        return expvx / np.sum(expvx, axis=-1, keepdims=True)

    def deriv(self, Y):
        return 1.

class Loss(object):
    pass

class MeanSquaredError(Loss):
    def compute(self, X, Y):
        return 0.5 * np.sum((X - Y) ** 2) / X.shape[0]

    def deriv(self, X, Y):
        return (X - Y)

class CrossEntropy(Loss):
    def compute(self, X, Y):
        return -np.dot(np.array(Y).flatten(), np.log(X.flatten())) / X.shape[0]

    def deriv(self, X, Y):
        return (X - Y)
