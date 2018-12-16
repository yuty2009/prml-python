# -*- coding: utf-8 -*-

import gzip
import pickle
import numpy as np
from basic.classifier import *


def main():

    f = gzip.open('E:\\prmldata\\mnist\\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    X1 = np.concatenate((training_data[0], validation_data[0]), axis=0)
    y1 = np.concatenate((training_data[1], validation_data[1]), axis=0)
    W = softmax_train(y1, X1)

    X21 = test_data[0]
    y21 = test_data[1]
    y22, dummy = softmax_predict(X21, W)
    acc = np.mean(np.equal(y22, y21).astype(np.float32))
    print(acc)


if __name__ == "__main__":
    main()