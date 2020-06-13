# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from utils.cifarreader import *
from deeplearning.cnn.keras.cifarcnn import *

imsize = 32
datapath = 'e:/prmldata/cifar-10/python'
cifar = CIFARReader(datapath, num_classes = 10)
trainset = cifar.get_train_dataset(onehot_label=True,
                                   reshape=True, new_shape=(-1, 3, imsize, imsize),
                                   tranpose=True, new_pos=(0, 2, 3, 1))
testset = cifar.get_test_dataset(onehot_label=True,
                                 reshape=True, new_shape=(-1, 3, imsize, imsize),
                                 tranpose=True, new_pos=(0, 2, 3, 1))
label_names = cifar.get_label_names()
X_train, y_train = trainset.images, trainset.labels
X_test, y_test = testset.images, testset.labels

#Visualizing CIFAR 10
n = 10
figure = np.zeros((imsize * n, imsize * n, 3))
for j in range(n):
    for k in range(n):
        i = np.random.choice(range(len(X_train)))
        figure[j * imsize: (j + 1) * imsize,
        k * imsize: (k + 1) * imsize, :] = X_train[i,:,:,:]
plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

model = CIFARCNN(input_shape=(imsize, imsize, 3), num_classes=10).model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size = 64, epochs = 50)

score = model.evaluate(X_test, y_test)
print("Total loss on Testing Set:", score[0])
print("Accuracy of Testing Set:", score[1])
