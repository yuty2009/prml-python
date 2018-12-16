# -*- coding: utf-8 -*-

import tensorflow as tf

def weight_bias(weight_shape, bias_shape):
    # Create variable named 'weights'.
    weights = tf.get_variable('weights', weight_shape,
        initializer=tf.random_normal_initializer(stddev=0.1))
    # Create variable named 'biases'.
    biases = tf.get_variable('biases', bias_shape,
        initializer=tf.constant_initializer(0))
    return weights, biases

def lm(vocabulary_size, embedding_size, num_sampled=64):
    with tf.variable_scope('input'):
        X = tf.placeholder(tf.int32, [None], name='X')
        ytrue = tf.placeholder(tf.int32, [None, 1], name='y')
        lr = tf.placeholder(tf.float32, name='lr')

    # Variables.
    embeddings = tf.get_variable(
        'embeddings', shape=[vocabulary_size, embedding_size],
         initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))

    with tf.variable_scope('output'):
        softmax_weights, softmax_biases = weight_bias(
            weight_shape=[vocabulary_size, embedding_size],
            bias_shape=[vocabulary_size])

    embeddings_X = tf.nn.embedding_lookup(embeddings, X)
    # Compute the softmax loss, using a sample of the negative labels each time.
    loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(
            weights=softmax_weights, biases=softmax_biases, inputs=embeddings_X,
            labels=ytrue, num_sampled=num_sampled, num_classes=vocabulary_size))

    train_op = tf.train.AdagradOptimizer(1.0).minimize(loss)

    return loss, train_op, embeddings, X, ytrue, lr