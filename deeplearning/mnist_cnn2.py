# -*- coding: utf-8 -*-

from utils.mnistreader import *
from deeplearning.cnn.keras.lenet5 import *

imsize = 28
datapath = 'e:/prmldata/mnist/'
mnist = MNISTReader(datapath=datapath)
trainset = mnist.get_train_dataset(onehot_label=True,
                                   reshape=True, new_shape=(-1, imsize, imsize, 1))
testset = mnist.get_test_dataset(onehot_label=True,
                                 reshape=True, new_shape=(-1, imsize, imsize, 1))
X_train, y_train = trainset.images, trainset.labels
X_test, y_test = testset.images, testset.labels

input_shape = X_train.shape[1:]
model = LeNet5(input_shape).model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size = 128, epochs = 20)

# y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test)
print("Total loss on Testing Set:", score[0])
print("Accuracy of Testing Set:", score[1])
