# -*- coding: utf-8 -*-
# reference: https://github.com/wuziheng/CNN-Numpy/blob/master/layers/base_conv.py

from .base import *
from ..activation import *

def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    for i in range(0, image.shape[1] - ksize[0] + 1, stride):
        for j in range(0, image.shape[2] - ksize[1] + 1, stride):
            col = image[:, i:i + ksize[0], j:j + ksize[1], :].reshape([-1])
            image_col.append(col)
    image_col = np.array(image_col)
    return image_col

class Conv2D(AbstractLayer):
    def __init__(self, inshape, wshape, bshape, stride=1, padding='valid', activation=Linear(), weight_init=weight_init, bias_init=bias_init):
        self.learnable = True
        self.inshape = inshape
        self.batchsize = inshape[0]
        self.wshape = wshape # [K K in_C out_C]
        self.bshape = bshape # [out_C]
        self.stride = stride
        self.padding = padding # 'valid' default or 'same'
        self.activation = activation
        self.weights = weight_init(self.wshape)
        self.biases = bias_init(self.bshape)

        self.inchannels = inshape[-1]
        self.outchannels = wshape[-1]
        self.ksize = wshape[0:-2]

        if padding.lower() == 'valid':
            self.outshape = (inshape[0],
                             int((inshape[1] - self.ksize[0]) / self.stride) + 1,
                             int((inshape[2] - self.ksize[1]) / self.stride) + 1,
                             self.outchannels)
        elif padding.lower() == 'same':
            self.outshape = (inshape[0],
                             int(inshape[1] / self.stride),
                             int(inshape[2] / self.stride),
                             self.outchannels)

        self.weights_grads = np.zeros(self.weights.shape)
        self.biases_grads = np.zeros(self.biases.shape)

    def forward(self, X):
        X = X.reshape(self.inshape)
        col_weights = self.weights.reshape([-1, self.outchannels])
        if self.padding.lower() == 'same':
            X = np.pad(X, ((0, 0), (int(self.ksize[0]/2), int(self.ksize[1]/2)),
                           (int(self.ksize[0]/2), int(self.ksize[1]/2)), (0, 0)),
                       'constant', constant_values=0)
        self.col_image = []
        Z = np.zeros(self.outshape)
        for i in range(self.batchsize):
            image_i = X[i][np.newaxis, :]
            col_image_i = im2col(image_i, ksize=self.ksize, stride=self.stride)
            Z[i] = np.reshape(np.dot(col_image_i, col_weights) + self.biases, self.outshape[1:])
            self.col_image.append(col_image_i)
        self.col_image = np.array(self.col_image)
        return self.activation.compute(Z)

    def backward(self, delta, A):
        delta = np.reshape(delta, A.shape)
        delta *= self.activation.deriv(A)
        col_delta = np.reshape(delta, [self.batchsize, -1, self.outchannels])

        for i in range(self.batchsize):
            self.weights_grads += np.dot(self.col_image[i].T, col_delta[i]).reshape(self.weights.shape)
        self.weights_grads /= self.batchsize
        self.biases_grads = np.mean(col_delta, axis=(0, 1))

        # deconv of padded delta with flippd kernel to get next_delta
        if self.padding.lower() == 'valid':
            pad_delta = np.pad(delta, ((0, 0), (self.ksize[0]-1, self.ksize[1]-1),
                                       (self.ksize[0]-1, self.ksize[1]-1), (0, 0)),
                               'constant', constant_values=0)
        elif self.padding.lower() == 'same':
            pad_delta = np.pad(delta, ((0, 0), (int(self.ksize[0]/2), int(self.ksize[1]/2)),
                                       (int(self.ksize[0]/2), int(self.ksize[1]/2)), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.inchannels])
        col_pad_delta = np.array(
            [im2col(pad_delta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_delta = np.dot(col_pad_delta, col_flip_weights)
        next_delta = np.reshape(next_delta, self.inshape)
        return next_delta

    def applygrad(self, lr=1e-4, wd=4e-4):
        self.weights *= (1 - wd)
        self.biases *= (1 - wd)
        self.weights -= lr * self.weights_grads
        self.biases -= lr * self.biases_grads
        self.weights_grads = np.zeros(self.weights_grads.shape)
        self.biases_grads = np.zeros(self.biases_grads.shape)