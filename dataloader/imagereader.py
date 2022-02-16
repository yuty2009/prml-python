# -*- coding: utf-8 -*-

import os
import io
import glob
import numpy as np
from PIL import Image


def load_imagedata(imagepath, scale=True, vectorize=True, grayscale=True,
                   resize=False, new_size=None,
                   transpose=False, new_pos=(0, 1, 2, 3)):
    imagedata = []
    # in-memory bytes of image file content
    if isinstance(imagepath, bytes):
        img = read_image(imagepath, resize=resize, new_size=new_size, grayscale=grayscale)
        imagedata.append(img)
    # ndarray imagelist
    elif not isinstance(imagepath, str):
        for filepath in imagepath:
            img = read_image(filepath, resize=resize, new_size=new_size, grayscale=grayscale)
            imagedata.append(img)
    # one single imagepath
    elif os.path.isfile(imagepath):
        img = read_image(imagepath, resize=resize, new_size=new_size, grayscale=grayscale)
        imagedata.append(img)
    # image directory
    elif os.path.isdir(imagepath):
        for filepath in glob.glob(imagepath + "/*.jpg"):
            img = read_image(filepath, resize=resize, new_size=new_size, grayscale=grayscale)
            imagedata.append(img)

    imagedata = np.asarray(imagedata)

    if grayscale:
        imagedata = np.expand_dims(imagedata, 3)
    if vectorize:
        imagedata = imagedata.reshape(imagedata.shape[0], np.prod(imagedata.shape[1:]))
    if scale:
        imagedata = imagedata.astype(np.float32) / 255.0
    if transpose:
        imagedata = imagedata.transpose(new_pos)

    return imagedata


def read_image(imagepath, resize=False, new_size=None, grayscale=True):
    if isinstance(imagepath, bytes):
        thepath = io.BytesIO()
        thepath.write(imagepath)
        thepath.seek(0)
    else:
        thepath = imagepath

    if grayscale:
        im = Image.open(thepath).convert('L')
    else:
        im = Image.open(thepath)
    if resize:
        im = im.resize(tuple(new_size), Image.BILINEAR)
    return np.asarray(im)


def onehot_labels(labels, num_classes):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_onehot = np.zeros((num_labels, num_classes))
    labels_onehot.flat[index_offset + labels.ravel()] = 1
    return labels_onehot


def show(image):
    if isinstance(image, str):
        im = Image.open(image)
        img = np.asarray(im)
    else:
        img = image

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(img.squeeze())
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    if isinstance(image, str): plt.title(image)
    plt.show()


def ascii_show(image):
    if isinstance(image, str):
        im = Image.open(image).convert('L')
        img = np.asarray(im)
    else:
        img = image

    for y in img:
        row = ""
        for x in y:
            row += '{0: <4}'.format(x)
        print(row)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class DataSet(object):
    def __init__(self, images, labels,
                 scale=True, vectorize=True, grayscale=True,
                 resize=False, new_size=None,
                 transpose=False, new_pos=(0, 1, 2, 3),
                 readonload=False, readalready=False):

        assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))

        self._num_examples = images.shape[0]
        self._shape = images.shape

        self.scale = scale
        self.vectorize = vectorize
        self.grayscale = grayscale
        self.resize = resize
        self.new_size = new_size
        self.transpose = transpose
        self.new_pos = new_pos
        self.readonload = readonload
        self.readalready = readalready

        if (not self.readalready) and self.readonload:
            self._images = load_imagedata(images, scale=scale, vectorize=vectorize, grayscale=grayscale,
                                          resize=resize, new_size=new_size,
                                          transpose=transpose, new_pos=new_pos)
            self.readalready = True
        else:
            self._images = images
        self._labels = labels

        self._indices = np.arange(self._num_examples)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_folds = 10  # 10-fold cross-validation default
        self._num_examples_fold = self._num_examples // self._num_folds
        self._folds_completed = 0
        self._fold = 0

    @property
    def images(self):
        return self._images

    @property
    def shape(self):
        return self._shape

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

    def create(self, images, labels):
        return DataSet(images, labels,
                       scale=self.scale, vectorize=self.vectorize, grayscale=self.grayscale,
                       resize=self.resize, new_size=self.new_size,
                       transpose=self.transpose, new_pos=self.new_pos,
                       readonload=self.readonload, readalready=self.readalready)

    def shuffle(self):
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._indices = perm

    def get_portiondata(self, indices):
        if self.readalready:
            subimages = self._images[indices]
        else:
            subimages = load_imagedata(self._images[indices],
                                       scale=self.scale, vectorize=self.vectorize, grayscale=self.grayscale,
                                       resize=self.resize, new_size=self.new_size,
                                       transpose=self.transpose, new_pos=self.new_pos)
        sublabels = self._labels[indices]
        return subimages, sublabels

    def get_subset(self, ratio, shuffle=True):
        ratio = ratio / np.sum(ratio)
        num_total = self.num_examples
        num_each = (num_total * ratio).astype(int)
        ends = np.cumsum(num_each)
        ends[-1] = num_total
        starts = np.copy(ends)
        starts[1:] = starts[0:-1]
        starts[0] = 0
        if shuffle: self.shuffle()
        subsets = []
        for (start, end) in (starts, ends):
            subimages, sublabels = self.get_portiondata(self._indices[start:end])
            subset = self.create(subimages, sublabels)
            subsets.append(subset)
        return subsets

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
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
