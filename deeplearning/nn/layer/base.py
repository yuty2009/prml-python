# -*- coding: utf-8 -*-

import numpy as np
from abc import ABCMeta, abstractmethod
from ..activation import *

def zero_init(var_shape):
    return np.zeros(var_shape)

def const_init(var_shape, const_value):
    return const_value*np.ones(var_shape)

def weight_init(weight_shape, weight_value=None):
    if weight_value is None:
        weights = np.random.normal(0, 0.1, weight_shape)
    else:
        if np.isscalar(weight_value):
            weights = weight_value*np.ones(weight_shape)
        else:
            weights = weight_value
    return weights

def bias_init(bias_shape, bias_value=None):
    if bias_value is None:
        biases = np.random.normal(0, 0.1, bias_shape)
    else:
        if np.isscalar(bias_value):
            biases = bias_value*np.ones(bias_shape)
        else:
            biases = bias_value
    return biases

class AbstractLayer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def backward(self, delta, A):
        raise NotImplementedError()

    @abstractmethod
    def applygrad(self, lr, wd):
        raise NotImplementedError()