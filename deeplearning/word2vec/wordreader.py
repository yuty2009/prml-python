# -*- coding: utf-8 -*-

import zipfile
import numpy as np
from collections import Counter

UNKNOWN_FLAG = 'UNKNOWN'

def load_words(datapath):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(datapath) as f:
        words = f.read(f.namelist()[0]).decode().split()
    return words

def create_vocabulary(words, vocab_size=None):
    if vocab_size == None:
        vocab_counter = Counter(words).most_common(vocab_size)
    else:
        vocab_size = vocab_size - 1
        vocab_counter = Counter(words).most_common(vocab_size)
        vocab_counter = [[UNKNOWN_FLAG, -1]] + vocab_counter
    vocabulary, _ = zip(*vocab_counter)
    vocab_dict = dict(zip(vocabulary, range(len(vocabulary))))
    vocab_rdict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    return vocab_dict, vocab_rdict, vocab_counter

def word2index(words, vocab_size):
    vocab_dict, vocab_rdict, vocab_counter = create_vocabulary(words, vocab_size)
    count_unknown = 0
    words_indices = list()
    for word in words:
        if word in vocab_dict:
            index = vocab_dict[word]
        else:
            index = 0  # vocab_dict['UNKNOWN']
            count_unknown = count_unknown + 1
        words_indices.append(index)
    vocab_counter[0][1] = count_unknown
    return words_indices, vocab_dict, vocab_rdict, vocab_counter

class Dataset(object):
    def __init__(self, data, window=1):
        self._data = data
        self._window = window
        self._num_examples = len(data)
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def window(self):
        return self._window

    def set_window(self, window):
        self._window = window

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch_skipgram(self, batch_size):
        assert batch_size % (2*self.window) == 0
        batch_X = []
        batch_y = []
        for i in range(batch_size//(2*self.window)):
            for j in range(2*self.window):
                batch_X.append(self._data[self._index_in_epoch])
            if self._index_in_epoch < self.window:
                padding = self.window-self._index_in_epoch
                for j in range(self.window):
                    if j < padding:
                        batch_y.append(0)
                    else:
                        batch_y.append(self._data[self._index_in_epoch-j])
                    batch_y.append(self._data[self._index_in_epoch + self.window - j])
            elif self._index_in_epoch >= self._num_examples - self.window:
                padding = self.window + self._index_in_epoch + 1 - self._num_examples
                for j in range(self.window):
                    if j < padding:
                        batch_y.append(0)
                    else:
                        batch_y.append(self._data[self._index_in_epoch+j])
                    batch_y.append(self._data[self._index_in_epoch - self.window + j])
            else:
                for j in range(self.window):
                    batch_y.append(self._data[self._index_in_epoch - self.window + j])
                    batch_y.append(self._data[self._index_in_epoch + self.window - j])
            self._index_in_epoch = (self._index_in_epoch + 1) % self._num_examples
        return np.asarray(batch_X), np.reshape(batch_y, [-1, 1])

    def next_batch_cbow(self, batch_size, shuffle=True):
        pass