# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

def loadmodel(modelpath, bottleneck_tensor_name, input_tensor_name):
    with tf.Session() as sess:
        with gfile.FastGFile(modelpath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, input_tensor = (
                tf.import_graph_def(graph_def, name='',
                    return_elements=[bottleneck_tensor_name, input_tensor_name]))
    return sess.graph, bottleneck_tensor, input_tensor

def getbottlenecks(sess, imagedata, input_tensor, bottleneck_tensor):
    bottlenecks = sess.run(bottleneck_tensor, {input_tensor: imagedata})
    bottlenecks = np.squeeze(bottlenecks)
    return bottlenecks

def savebottlenecks(savepath, bottlenecks, labels, imagelist):
    np.savez(savepath, bottlenecks=bottlenecks, labels=labels, imagelist=imagelist)

def loadbottlenecks(savepath):
    npz = np.load(savepath)
    return (npz['bottlenecks'], npz['labels'], npz['imagelist'])

def finallayer(outlen, bottleneck_size):
    with tf.variable_scope('bottleneck_input'):
        X = tf.placeholder(tf.float32, shape=[None, bottleneck_size], name='X')
        ytrue = tf.placeholder(tf.float32, [None, outlen], name='y')
        lr = tf.placeholder(tf.float32, name='lr')

    with tf.variable_scope('output'):
        weights = tf.get_variable('weights', [bottleneck_size, outlen],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [outlen],
                                 initializer=tf.constant_initializer(0.1))
        ypred = tf.matmul(X, weights) + biases
        ypred = tf.identity(ypred, name='y')

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=ytrue, logits=ypred))
    regularization = tf.nn.l2_loss(weights)
    loss = cross_entropy
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(ypred, 1), tf.argmax(ytrue, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return (train_step, cross_entropy, accuracy, ypred, X, ytrue, lr)