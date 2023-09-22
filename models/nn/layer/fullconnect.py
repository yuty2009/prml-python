# -*- coding: utf-8 -*-

from .base import *
from ..activation import *

class FullConnect(AbstractLayer):
    def __init__(self, inshape, wshape, bshape, activation=Linear(), dropout=0, weight_init=weight_init, bias_init=bias_init):
        self.learnable = True
        self.inshape = inshape
        self.batchsize = inshape[0]
        self.wshape = wshape  # [in_Nodes out_Nodes]
        self.bshape = bshape  # [out_Nodes]
        self.dropout = dropout
        self.activation = activation
        self.weights = weight_init(self.wshape)
        self.biases = bias_init(self.bshape)

        self.outshape = [self.batchsize, wshape[1]]
        self.weights_grads = np.zeros(self.weights.shape)
        self.biases_grads = np.zeros(self.biases.shape)

    def forward(self, X):
        self.X = X.reshape([self.batchsize, -1])
        Z = np.dot(self.X, self.weights) + self.biases
        return self.activation.compute(Z)

    def backward(self, delta, A):
        delta *= self.activation.deriv(A)
        self.weights_grads = np.dot(self.X.T, delta) / self.batchsize
        self.biases_grads = np.mean(delta, axis=0)
        next_delta = np.dot(delta, self.weights.T)
        return next_delta

    def applygrad(self, lr=1e-4, wd=4e-4):
        self.weights *= (1. - wd)
        self.biases *= (1. - wd)
        self.weights -= lr * self.weights_grads
        self.biases -= lr * self.biases_grads
        self.weights_grads = np.zeros(self.weights_grads.shape)
        self.biases_grads = np.zeros(self.biases_grads.shape)