# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    
    f: backbone + projection mlp
    h: prediction mlp

    for x in loader: # load a minibatch x with n samples
        x1, x2 = aug(x), aug(x) # random augmentation
        z1, z2 = f(x1), f(x2) # projections, n-by-d
        p1, p2 = h(z1), h(z2) # predictions, n-by-d
        L = D(p1, z2)/2 + D(p2, z1)/2 # loss
        L.backward() # back-propagate
        update(f, h) # SGD update

    def D(p, z): # negative cosine similarity
        z = z.detach() # stop gradient
        p = normalize(p, dim=1) # l2-normalize
        z = normalize(z, dim=1) # l2-normalize
        return -(p*z).sum(dim=1).mean()
    """
    def __init__(self, encoder, encoder_dim=2048, feature_dim=2048, dim=512, num_mlplayers=2):
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
        if num_mlplayers == 2:
            self.projector = nn.Sequential(nn.Linear(encoder_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim, affine=False)) # output layer
        elif num_mlplayers == 3:
            self.projector = nn.Sequential(nn.Linear(encoder_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(feature_dim, feature_dim, bias=False),
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