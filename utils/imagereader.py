# -*- coding: utf-8 -*-

import os
import io
import glob
import numpy as np
from PIL import Image

def load_imagedata(imagepath, needresize=False, newsize=[],
                   dtype=np.float32, vectorize=True, grayscale=True):
    imagedata = []
    # in-memory bytes of image file content
    if isinstance(imagepath, bytes):
        img = read_image(imagepath, needresize=needresize, newsize=newsize, grayscale=grayscale)
        imagedata.append(img)
    # ndarray imagelist
    elif not isinstance(imagepath, str):
        for filepath in imagepath:
            img = read_image(filepath, needresize=needresize, newsize=newsize, grayscale=grayscale)
            imagedata.append(img)
    # one single imagepath
    elif os.path.isfile(imagepath):
        img = read_image(imagepath, needresize=needresize, newsize=newsize, grayscale=grayscale)
        imagedata.append(img)
    # image directory
    elif os.path.isdir(imagepath):
        for filepath in glob.glob(imagepath + "/*.jpg"):
            img = read_image(filepath, needresize=needresize, newsize=newsize, grayscale=grayscale)
            imagedata.append(img)

    imagedata = np.asarray(imagedata)

    if grayscale:
        imagedata = np.expand_dims(imagedata, 3)
    if vectorize:
        imagedata = imagedata.reshape(imagedata.shape[0], np.prod(imagedata.shape[1:]))
    if (imagedata.dtype != np.float32) and (dtype == np.float32):
        # Convert from [0, 255] -> [0.0, 1.0].
        imagedata = imagedata.astype(np.float32)
        imagedata = np.multiply(imagedata, 1.0 / 255.0)

    return imagedata

def read_image(imagepath, needresize=False, newsize=[], grayscale=True):
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
    if needresize:
        im = im.resize(tuple(newsize), Image.BILINEAR)
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

    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(img)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

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
    def __init__(self, images, labels, needresize=False, newsize=None,
                 dtype=np.float32, vectorize=True, grayscale=True,
                 readonload=False, readalready=False):

        if newsize is None:
            newsize = []
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._shape = images.shape

        self.needresize = needresize
        self.newsize = newsize
        self.dtype = dtype
        self.vectorize = vectorize
        self.grayscale = grayscale
        self.readonload = readonload
        self.readalready = readalready

        if (not self.readalready) and self.readonload:
            self._images = load_imagedata(images, needresize=needresize, newsize=newsize,
                                          dtype=dtype, vectorize=vectorize, grayscale=grayscale)
            self.readalready = True
        else:
            self._images = images
        self._labels = labels

        self._indices = np.arange(self._num_examples)
        self._epochs_completed = 0
        self._index_in_epoch = 0

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

    def reset(self):
        self._epochs_completed = 0

    def create(self, images, labels):
        return DataSet(images, labels,
                       needresize=self.needresize, newsize=self.newsize,
                       dtype=self.dtype, vectorize=self.vectorize, grayscale=self.grayscale,
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
                           needresize=self.needresize, newsize=self.newsize,
                           dtype=self.dtype, vectorize=self.vectorize, grayscale=self.grayscale)
        sublabels = self._labels[indices]
        return subimages, sublabels

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