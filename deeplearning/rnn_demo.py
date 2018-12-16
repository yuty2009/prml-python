# -*- coding: utf-8 -*-

import gzip
import pickle
import numpy as np
from nn.activation import *
from nn.network import *


## Generate one-hot coded labels
def onehot_labels(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot.flat[index_offset + labels.ravel()] = 1
    return labels_onehot


def main():

    nstep = 3
    batchsize = 50
    layers1 = [
        RNNCell([batchsize, nstep, 10], nstep, 100, 10, activation=Softmax())
        # LSTMCell([batchsize, nstep, 10], nstep, 100, 10, activation=Softmax())
        # GRUCell([batchsize, nstep, 10], nstep, 100, 10, activation=Softmax())
    ]
    loss = CrossEntropy()
    net1 = Network(layers1, loss=loss)

    inputs = np.random.randn(batchsize, nstep, 10)
    tmpouts1 = np.random.rand(batchsize, nstep, 10)
    indices = np.argmax(tmpouts1, axis=2)
    outputs1 = np.zeros(tmpouts1.shape)
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            outputs1[i, j, indices[i, j]] = 1

    ypred1 = net1.forward(inputs)

    net1.checkgrads(inputs, outputs1)


if __name__ == "__main__":
    main()