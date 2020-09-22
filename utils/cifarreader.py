# -*- coding: utf-8 -*-

import os
import pickle
import platform
import numpy as np
from .imagereader import *
# import matplotlib.pyplot as plt


class CIFARReader(object):
    def __init__(self, datapath, num_classes = 10):
        self.datapath = datapath
        self.num_classes = num_classes

    def get_train_dataset(self, onehot_label=False, scale=True,
                          reshape=False, new_shape=(-1, 32, 32, 3),
                          transpose=False, new_pos=(0, 1, 2, 3)):
        train_images = []
        train_labels = []
        for i in range(5):
            f_batch = os.path.join(self.datapath, 'data_batch_' + str(i+1))
            images_batch, labels_batch = self.get_batch_data(f_batch)
            train_images.append(images_batch)
            train_labels.append(labels_batch)
        train_images = np.concatenate(train_images, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        if scale:
            train_images = train_images.astype('float32') / 255.
        if reshape:
            train_images = np.reshape(train_images, new_shape)
        if transpose:
            train_images = train_images.transpose(new_pos)
        if onehot_label:
            train_labels = onehot_labels(train_labels, self.num_classes)
        dataset = DataSet(train_images, train_labels,
                          grayscale = False,
                          readalready = True)
        return dataset

    def get_test_dataset(self, onehot_label=False, scale=True,
                          reshape=False, new_shape=(-1, 32, 32, 3),
                          transpose=False, new_pos=(0, 1, 2, 3)):
        f_batch = os.path.join(self.datapath, 'test_batch')
        test_images, test_labels = self.get_batch_data(f_batch)
        if scale:
            test_images = test_images.astype('float32') / 255.
        if reshape:
            test_images = np.reshape(test_images, new_shape)
        if transpose:
            test_images = test_images.transpose(new_pos)
        if onehot_label:
            test_labels = onehot_labels(test_labels, self.num_classes)
        dataset = DataSet(test_images, test_labels,
                          grayscale=False,
                          readalready=True)
        return dataset

    def get_label_names(self):
        f_batch = os.path.join(self.datapath, 'batches.meta')
        dict = self.unpickle(f_batch)
        return np.array(dict['label_names'])

    def get_batch_data(self, f_batch):
        dict = self.unpickle(f_batch)
        images = np.array(dict['data'])
        labels = np.array(dict['labels'])
        return images, labels

    def unpickle(self, f_batch):
        with open(f_batch, 'rb') as fo:
            version = platform.python_version_tuple()
            if version[0] == '2':
                dict = pickle.load(fo, encoding='bytes')
            elif version[0] == '3':
                dict = pickle.load(fo, encoding='latin1')
        return dict