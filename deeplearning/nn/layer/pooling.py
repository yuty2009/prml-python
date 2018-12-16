# -*- coding: utf-8 -*-

from .base import *
from ..activation import *

class AvgPooling(object):
    def __init__(self, inshape, ksize=2, stride=2):
        self.learnable = False
        self.inshape = inshape
        self.ksize = ksize
        self.stride = stride
        self.outchannels = inshape[-1]
        self.integral = np.zeros(inshape)
        self.index = np.zeros(inshape)

    def forward(self, x):
        for b in range(x.shape[0]):
            for c in range(self.outchannels):
                for i in range(x.shape[1]):
                    row_sum = 0
                    for j in range(x.shape[2]):
                        row_sum += x[b, i, j, c]
                        if i == 0:
                            self.integral[b, i, j, c] = row_sum
                        else:
                            self.integral[b, i, j, c] = self.integral[b, i - 1, j, c] + row_sum

        out = np.zeros([x.shape[0], int(x.shape[1]/self.stride), int(x.shape[2]/self.stride), self.outchannels], dtype=float)

        # integral calculate pooling
        for b in range(x.shape[0]):
            for c in range(self.outchannels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        self.index[b, i:i + self.ksize, j:j + self.ksize, c] = 1
                        if i == 0 and j == 0:
                            out[b, int(i/self.stride), int(j/self.stride), c] = self.integral[
                                b, self.ksize - 1, self.ksize - 1, c]

                        elif i == 0:
                            out[b, int(i/self.stride), int(j/self.stride), c] = self.integral[b, 1, j + self.ksize - 1, c] - \
                                                                          self.integral[b, 1, j - 1, c]
                        elif j == 0:
                            out[b, int(i/self.stride), int(j/self.stride), c] = self.integral[b, i + self.ksize - 1, 1, c] - \
                                                                          self.integral[b, i - 1, 1, c]
                        else:
                            out[b, int(i/self.stride), int(j/self.stride), c] = self.integral[
                                                                              b, i + self.ksize - 1, j + self.ksize - 1, c] - \
                                                                          self.integral[
                                                                              b, i - 1, j + self.ksize - 1, c] - \
                                                                          self.integral[
                                                                              b, i + self.ksize - 1, j - 1, c] + \
                                                                          self.integral[b, i - 1, j - 1, c]

        out /= (self.ksize * self.ksize)
        return out

    def backward(self, delta, A):
        delta = np.reshape(delta, A.shape)
        next_delta = np.repeat(delta, self.stride, axis=1)
        next_delta = np.repeat(next_delta, self.stride, axis=2)
        next_delta = next_delta*self.index
        return next_delta/(self.ksize*self.ksize)


class MaxPooling(object):
    def __init__(self, inshape, ksize=2, stride=2):
        self.learnable = False
        self.inshape = inshape
        self.ksize = ksize
        self.stride = stride
        self.outchannels = inshape[-1]
        self.index = np.zeros(inshape)

    def forward(self, x):
        out = np.zeros([x.shape[0], int(x.shape[1]/self.stride), int(x.shape[2]/self.stride), self.outchannels], dtype=float)
        for b in range(x.shape[0]):
            for c in range(self.outchannels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        block = x[b, i:i + self.ksize, j:j + self.ksize, c]
                        value, index = np.max(block), np.argmax(block)
                        out[b, int(i / self.stride), int(j / self.stride), c] = value
                        self.index[b, i + int(np.floor(index / self.stride)), j + index % self.stride, c] = 1
        return out

    def backward(self, delta, A):
        delta = np.reshape(delta, A.shape)
        next_delta = np.repeat(delta, self.stride, axis=1)
        next_delta = np.repeat(next_delta, self.stride, axis=2)
        next_delta = next_delta * self.index
        return next_delta


class StocasticPooling(object):
    def __init__(self, inshape, ksize=2, stride=2):
        self.learnable = False
        self.inshape = inshape
        self.ksize = ksize
        self.stride = stride
        self.outchannels = inshape[-1]
        self.index = np.zeros(inshape)

    def forward(self, x):
        out = np.zeros([x.shape[0], int(x.shape[1]/self.stride), int(x.shape[2]/self.stride), self.outchannels], dtype=float)
        for b in range(x.shape[0]):
            for c in range(self.outchannels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        block = x[b, i:i + self.ksize, j:j + self.ksize, c]
                        index = np.random.randint(self.ksize*self.ksize)
                        value = block[int(np.floor(index/self.stride)), index%self.stride]
                        out[b, int(i/self.stride), int(j/self.stride), c] = value
                        self.index[b, i + int(np.floor(index/self.stride)), j + index%self.stride, c] = 1
        return out

    def backward(self, delta, A):
        delta = np.reshape(delta, A.shape)
        next_delta = np.repeat(delta, self.stride, axis=1)
        next_delta = np.repeat(next_delta, self.stride, axis=2)
        next_delta = next_delta * self.index
        return next_delta