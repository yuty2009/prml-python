# -*- coding: utf-8 -*-

from utils.mnistreader import *
from deeplearning.cnn.keras.lenet5 import *

imsize = 28
datapath = 'e:/prmldata/mnist/'
mnist = MNISTReader(datapath=datapath)
trainset = mnist.get_train_dataset(onehot_label=True)
testset = mnist.get_test_dataset(onehot_label=True)
X_train, y_train = trainset.images, trainset.labels
X_test, y_test = testset.images, testset.labels

num_classes = 10
input_shape = X_train.shape[1:]
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size = 128, epochs = 20)

# y_pred = model.predict(X_test)

score = model.evaluate(X_test, y_test)
print("Total loss on Testing Set:", score[0])
print("Accuracy of Testing Set:", score[1])
