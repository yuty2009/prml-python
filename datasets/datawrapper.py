# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio


def read_matdata(filepath, keys):
    data = {}
    f = sio.loadmat(filepath)
    # f = h5py.File(filepath, 'r')
    for key in keys:
        data[key] = f[key]
    return data


class Dataset(object):
    def __init__(self, features, labels):
        self._features = features
        self._labels = labels
        self._num_examples = self._features.shape[0]
        self._indices = np.arange(self._num_examples)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_folds = 10 # 10-fold cross-validation default
        self._num_examples_fold = self._num_examples // self._num_folds
        self._folds_completed = 0
        self._fold = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def folds_completed(self):
        return self._folds_completed

    def set_num_folds(self, num_folds):
        self._num_folds = num_folds
        self._num_examples_fold = self._num_examples // self._num_folds

    def reset(self):
        self._epochs_completed = 0

    def shuffle(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._indices = perm

    def get_portiondata(self, indices):
        return self._features[indices], self._labels[indices]

    def get_subset(self, ratio, shuffle=True):
        ratio = ratio / np.sum(ratio)
        num_total = self.num_examples
        num_each = (num_total * ratio).astype(int)
        ends = np.cumsum(num_each)
        ends[-1] = num_total
        starts = np.copy(ends)
        starts[1:]  = starts[0:-1]
        starts[0] = 0
        if shuffle: self.shuffle()
        subsets = []
        for (start, end) in (starts, ends):
            subfeatures, sublabels = self.get_portiondata(self._indices[start:end])
            subset = Dataset(subfeatures, sublabels)
            subsets.append(subset)
        return subsets

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set"""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if start == 0 and shuffle:
            self.shuffle()
        # Go to the next epoch
        if start + batch_size >= self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            self._index_in_epoch = 0
            end = self._num_examples
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
        indices_portion = self._indices[start:end]
        return self.get_portiondata(indices_portion)

    def next_fold(self, shuffle=True):
        """Generate train set and test set for K-fold cross-validation"""
        start = self._fold
        # Shuffle for the first epoch
        if start == 0 and shuffle:
            self.shuffle()
        indices_test = self._indices[self._fold * self._num_examples_fold:
                                     (self._fold + 1) * self._num_examples_fold]
        indices_train = np.setdiff1d(self._indices, indices_test)
        self._fold += 1
        if self._fold >= self._num_folds:
            self._fold = 0
        return self.get_portiondata(indices_train) + \
               self.get_portiondata(indices_test)
