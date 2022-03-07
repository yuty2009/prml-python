# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, encoder, encoder_dim=2048, feature_dim=2048, dim=512):
        """
        encoder: encoder you want to use to get feature representations (eg. resnet50)
        encoder_dim: dimension of the encoder output, your feature dimension (default: 2048 for resnets)
        feature_dim: dimension of the projector output (default: 2048)
        dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the online encoder
        self.encoder = encoder
        # create the online projector
        self.projector = nn.Sequential(nn.Linear(encoder_dim, encoder_dim, bias=False),
                                        nn.BatchNorm1d(encoder_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(encoder_dim, encoder_dim, bias=False),
                                        nn.BatchNorm1d(encoder_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(encoder_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim, affine=False)) # output layer
        self.model = nn.Sequential(self.encoder, self.projector)
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(feature_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim, feature_dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, t1, t2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        t1 = self.model(x1) # NxC
        t2 = self.model(x2) # NxC

        p1 = self.predictor(t1) # NxC
        p2 = self.predictor(t2) # NxC

        return p1, p2, t1.detach(), t2.detach()