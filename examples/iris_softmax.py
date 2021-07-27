# -*- coding: utf-8 -*-

import numpy as np
from common.pytorch import *
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():

    X, y = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = SoftmaxClassifier()
    W = model.fit(X_train, y_train)
    yp, dummy = model.predict(X_test)
    acc = np.mean(np.equal(yp, y_test).astype(np.float32))
    print(acc)


if __name__ == "__main__":
    main()
