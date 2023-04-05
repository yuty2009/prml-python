# refer to: https://zhuanlan.zhihu.com/p/152294675

import os
import numpy as np
import torch

import sys; sys.path.append(os.path.dirname(__file__)+"/../")
from mnistseq import MNISTSeqSet
from cpc import CPC, CPCLoss, CNNEncoder


datapath = '/Users/yuty2009/data/prmldata/mnist/MNIST/raw'
trainset = MNISTSeqSet(
    root=datapath, subset='train',
    image_size=64, color=True, rescale=False, terms=4, predict_terms=4,
)
