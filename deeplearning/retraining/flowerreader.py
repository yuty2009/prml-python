# -*- coding: utf-8 -*-

import os, glob
import numpy as np
from imagereader import *

def load_dataset(datapath, categories, needresize=False, newsize=[],
                 dtype=np.float32, vectorize=True, grayscale=True, readonload=False):

    images, labels = load_imagelist(datapath, categories)
    labels = onehot_labels(labels, len(categories))

    dataset = DataSet(images, labels, dtype=dtype, needresize=needresize, newsize=newsize,
        vectorize=vectorize, grayscale=grayscale, readonload=readonload)
    return dataset

def load_imagelist(imagepath, categories, fileext='jpg'):
    labels = []
    images = []
    num_classes = len(categories)
    for i in range(num_classes):
        label = i
        catepath = os.path.join(imagepath, categories[i])
        for filepath in glob.glob(catepath + '/*.' + fileext):
            labels.append(label)
            images.append(filepath)

    print("Read %d images from [%s]" % (len(labels), imagepath))
    return np.asarray(images), np.asarray(labels)

def load_bottlenecks(datapath, categories):
    bottlenecklist, labels = load_imagelist(datapath, categories, fileext='bottleneck')
    bottlenecks = load_bottleneckdata(bottlenecklist)
    labels = onehot_labels(labels, len(categories))

    dataset = DataSet(bottlenecks, labels, readalready=True)
    return dataset

def load_bottleneckdata(bottlenecklist):
    bottlenecks = []
    for bottleneckpath in bottlenecklist:
        with open(bottleneckpath, 'r') as fp:
            bottleneck_string = fp.read()
            bottleneck = [float(x) for x in bottleneck_string.split(',')]
            bottlenecks.append(bottleneck)
    return np.asarray(bottlenecks)
