# -*- coding: utf-8 -*-

from .base import *

class Reshape(AbstractLayer):
    def __init__(self, newshape):
        self.learnable = False
        self.outshape = newshape

    def forward(self, X):
        return X.reshape(self.outshape)

    def backward(self, delta, A):
        return delta # delta.reshape(self.outshape)

    def applygrad(self, lr=1e-4, wd=4e-4):
        pass