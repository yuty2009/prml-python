# -*- coding: utf-8 -*-

import gzip
import pickle
import numpy as np
import tensorflow as tf
from deeplearning.nn.activation import *
from deeplearning.nn.network import *


## Generate one-hot coded labels
def onehot_labels(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot.flat[index_offset + labels.ravel()] = 1
    return labels_onehot


def tffeedforward(input, weights, biases):
    w1, b1 = weights[0], biases[0]
    w2, b2 = weights[1], biases[1]
    w3, b3 = weights[2], biases[2]
    w4, b4 = weights[3], biases[3]
    o1 = tf.nn.sigmoid(tf.nn.conv2d(input, w1, strides=[1, 1, 1, 1], padding='SAME') + b1)
    o2 = tf.nn.max_pool(o1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    o3 = tf.nn.sigmoid(tf.nn.conv2d(o2, w2, strides=[1, 1, 1, 1], padding='SAME') + b2)
    o4 = tf.nn.max_pool(o3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    o44 = tf.reshape(o4, [-1, w3.shape[0]])
    o5 = tf.nn.sigmoid(tf.matmul(o44, w3) + b3)
    o6 = tf.nn.softmax(tf.matmul(o5, w4) + b4)
    return o6


def main():

    # f = gzip.open('E:\\prmldata\\mnist\\mnist.pkl.gz', 'rb')
    # training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    # f.close()

    K = 32
    L = 64
    batchsize = 50
    layers = [
        Conv2D([batchsize, 28, 28, 1], [5, 5, 1, K], [K], padding='same', activation=Sigmoid()),
        MaxPooling([batchsize, 28, 28, K], ksize=2, stride=2),
        Conv2D([batchsize, 14, 14, K], [5, 5, K, L], [L], padding='same', activation=Sigmoid()),
        MaxPooling([batchsize, 14, 14, L], ksize=2, stride=2),
        FullConnect([batchsize, 7*7*L], [7*7*L, 1000], [1000], activation=Sigmoid()),
        FullConnect([batchsize, 1000], [1000, 10], [10], activation=Softmax())
    ]
    loss = CrossEntropy()
    net1 = Network(layers, loss=loss)

    inputs = np.random.rand(batchsize,28,28,1)
    outputs = np.random.rand(batchsize,10)
    indices = np.argmax(outputs, axis=1)
    tmp = np.zeros(outputs.shape)
    for i in range(len(indices)):
       tmp[i, indices[i]] = 1
    outputs = tmp
    ypred1 = net1.forward(inputs)

    net1.checkgrads(inputs, outputs)

    weigths = [layers[0].weights, layers[2].weights, layers[4].weights, layers[5].weights]
    biases = [layers[0].biases, layers[2].biases, layers[4].biases, layers[5].biases]
    ynet2 = tffeedforward(inputs, weigths, biases)
    sess = tf.Session()
    ypred2 = sess.run(ynet2)

    lr = 0.001
    inputs = training_data[0][0:batchsize, :]
    outputs = onehot_labels(training_data[1][0:batchsize], 10)


if __name__ == "__main__":
    main()