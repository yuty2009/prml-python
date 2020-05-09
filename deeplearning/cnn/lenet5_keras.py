# -*- coding: utf-8 -*-
#

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class LeNet5:
    def __init__(self, input_shape=None, num_classes=10):
        super(LeNet5, self).__init__()
        if input_shape is None:
            input_shape = (32, 32, 1)
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        # self.model.add(MaxPooling2D((2,2),strides=(2,2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2,2),strides=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        # self.model.add(Dense(84, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
