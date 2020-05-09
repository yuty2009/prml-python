# -*- coding: utf-8 -*-
#

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


class CIFARCNN:
    def __init__(self, input_shape, num_classes=10):
        super(CIFARCNN, self).__init__()

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
                         input_shape=input_shape))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))
