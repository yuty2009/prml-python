# -*- coding: utf-8 -*-
#

import torch
import struct
import numpy as np
from torch.utils.data import Dataset

datapath = 'e:/prmldata/mnist/'
F_TRAIN_IMAGES = 'train-images-idx3-ubyte'
F_TRAIN_LABELS = 'train-labels-idx1-ubyte'
F_TEST_IMAGES = 't10k-images-idx3-ubyte'
F_TEST_LABELS = 't10k-labels-idx1-ubyte'


class MNISTTrainset(Dataset):
    def __init__(self):
        reader = MNISTReader()
        self.X = reader.read_raw_images(datapath + F_TRAIN_IMAGES)
        self.y = reader.read_raw_labels(datapath + F_TRAIN_LABELS)
        self.X = self.X.astype('float32') / 255.
        self.X = np.reshape(self.X, (-1, 28, 28, 3))
        self.X = self.X.transpose((0, 3, 1, 2))
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)
        self.len = len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


class MNISTTestset(Dataset):
    def __init__(self):
        reader = MNISTReader()
        self.X = reader.read_raw_images(datapath + F_TEST_IMAGES)
        self.y = reader.read_raw_labels(datapath + F_TEST_LABELS)
        self.X = self.X.astype('float32') / 255.
        self.X = np.reshape(self.X, (-1, 28, 28, 3))
        self.X = self.X.transpose((0, 3, 1, 2))
        self.X = torch.from_numpy(self.X)
        self.y = torch.from_numpy(self.y)
        self.len = len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


class MNISTReader:
    def __init__(self):
        self._tag = '>'
        self._twoBytes = 'II'
        self._fourBytes = 'IIII'
        self._pictureBytes = '784B'
        self._labelByte = '1B'
        self._twoBytes2 = self._tag + self._twoBytes
        self._fourBytes2 = self._tag + self._fourBytes
        self._pictureBytes2 = self._tag + self._pictureBytes
        self._labelByte2 = self._tag + self._labelByte

    def read_raw_images(self, filepath):
        fp = open(filepath, 'rb')
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