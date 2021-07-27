# -*- coding: utf-8 -*-

import gzip
import pickle
import numpy as np
from common.linear import *


def main():

    f = gzip.open('e:/prmldata/mnist/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    X1 = np.concatenate((training_data[0], validation_data[0]), axis=0)
    y1 = np.concatenate((training_data[1], validation_data[1]), axis=0)

    model = SoftmaxClassifier()
    W = model.fit(X1, y1)

    X21 = test_data[0]
    y21 = test_data[1]
    y22, dummy = model.predict(X21)
    acc = np.mean(np.equal(y22, y21).astype(np.float32))
    print(acc)


if __name__ == "__main__":
    main()
