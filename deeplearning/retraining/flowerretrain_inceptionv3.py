# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import flowerreader as reader
import retrainmodel as model

INPUT_WIDTH = 299
INPUT_HEIGHT = 299
INPUT_DEPTH = 3
INPUT_TENSOR_NAME = 'ResizeBilinear:0'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
CATEGORIES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

def main():

    datapath = 'e:\\prmldata\\tensorflow\\flower_photos'
    modelpath = os.path.join(datapath, 'models/')

    bottleneckset = reader.load_bottlenecks(datapath, CATEGORIES)
    train_bn, test_bn = bottleneckset.get_subset([80, 20])

    train_step, cross_entropy, accuracy, ypred, X, ytrue, lr = \
        model.finallayer(len(CATEGORIES), BOTTLENECK_TENSOR_SIZE)

    maxstep = 20000
    reportstep = 100
    savestep = 10000
    lr_start = 1e-3
    batch_size = 100

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        for i in range(maxstep):
            batch_X, batch_y = train_bn.next_batch(batch_size)
            lr_now = lr_start * (1 + 1e-4 * i) ** (-0.75)
            if i % reportstep == 0:
                train_loss = cross_entropy.eval(feed_dict={
                    X: batch_X, ytrue: batch_y})
                train_accuracy = accuracy.eval(feed_dict={
                    X: batch_X, ytrue: batch_y})
                print('Step=%d, lr=%.4f, loss=%.4f, train accuracy=%g'
                      % (i, lr_now, train_loss, train_accuracy))
            if (i + 1) % savestep == 0:
                saver.save(sess, modelpath+'flower-retrained-model', global_step=i + 1)
            train_step.run(feed_dict={X: batch_X, ytrue: batch_y, lr: lr_now})

        batch_size = 1
        batch_num = int(test_bn.num_examples / batch_size)
        test_accuracy = 0
        for i in range(batch_num):
            batch_X, batch_y = test_bn.next_batch(batch_size, shuffle=False)
            test_accuracy += accuracy.eval(
                feed_dict={X: batch_X, ytrue: batch_y})
        test_accuracy /= batch_num
        print("test accuracy %g" % test_accuracy)


if __name__ == "__main__":
    main()