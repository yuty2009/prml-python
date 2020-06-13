# -*- coding: utf-8 -*-

import os
import struct
import numpy as np
from .imagereader import *
# import matplotlib.pyplot as plt


class MNISTReader(object):
    def __init__(self, f_train_images, f_train_labels,
                 f_test_images, f_test_labels,
                 num_classes = 10):
        self.num_classes = num_classes
        self._f_train_images = f_train_images
        self._f_train_labels = f_train_labels
        self._f_test_images = f_test_images
        self._f_test_labels = f_test_labels

        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def get_train_dataset(self, onehot_label=False, scale=True,
                          normalize=False, mean_std=(0.5, 0.5),
                          reshape=False, new_shape=(-1, 28, 28, 1),
                          tranpose=False, new_pos=(0, 1, 2, 3)):
        train_images = self.read_raw_images(self._f_train_images)
        train_labels = self.read_raw_labels(self._f_train_labels)
        if scale:
            train_images = train_images.astype('float32') / 255.
        if normalize:
            train_images = (train_images.astype('float32') - mean_std[0]) / mean_std[1]
        if reshape:
            train_images = np.reshape(train_images, new_shape)
        if tranpose:
            train_images = train_images.transpose(new_pos)
        if onehot_label:
            train_labels = onehot_labels(train_labels, self.num_classes)
        dataset = DataSet(train_images, train_labels, readalready=True)
        return dataset

    def get_test_dataset(self, onehot_label=False, scale=True,
                         normalize=False, mean_std=(0.5, 0.5),
                         reshape=False, new_shape=(-1, 28, 28, 1),
                         tranpose=False, new_pos=(0, 1, 2, 3)):
        test_images = self.read_raw_images(self._f_test_images)
        test_labels = self.read_raw_labels(self._f_test_labels)
        if scale:
            test_images = test_images.astype('float32') / 255.
        if normalize:
            test_images = (test_images.astype('float32') - mean_std[0]) / mean_std[1]
        if reshape:
            test_images = np.reshape(test_images, new_shape)
        if tranpose:
            test_images = test_images.transpose(new_pos)
        if onehot_label:
            test_labels = onehot_labels(test_labels, self.num_classes)
        dataset = DataSet(test_images, test_labels, readalready=True)
        return dataset

    def read_raw_images(self, filepath):
        fp = open(filepath, 'rb')  # 以二进制方式打开文件
        data = fp.read()
        fp.close()
        index = 0
        numMagic, numImgs, numRows, numCols = struct.unpack_from(self._fourBytes2,
                                                                 data,
                                                                 index)
        index += struct.calcsize(self._fourBytes)
        images = []
        for i in range(numImgs):
            imgVal = struct.unpack_from(self._pictureBytes2, data, index)
            index += struct.calcsize(self._pictureBytes2)
            imgVal = list(imgVal)
            images.append(imgVal)
        return np.array(images)

    def read_raw_labels(self, filepath):
        fp = open(filepath, 'rb')
        data = fp.read()
        fp.close()
        index = 0
        magic, numItems = struct.unpack_from(self._twoBytes2, data, index)
        index += struct.calcsize(self._twoBytes2)
        labels = []
        for x in range(numItems):
            im = struct.unpack_from(self._labelByte2, data, index)
            index += struct.calcsize(self._labelByte2)
            labels.append(im[0])
        return np.array(labels)
