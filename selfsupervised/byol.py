# Copy from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
import copy
import torch
import torch.nn as nn


class BYOL(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, encoder, encoder_dim=2048, feature_dim=2048, dim=512, m=0.999):
        """
        encoder: encoder you want to use to get feature representations (eg. resnet50)
        encoder_dim: dimension of the encoder output, your feature dimension (default: 2048 for resnets)
        feature_dim: dimension of the projector output (default: 2048)
        dim: hidden dimension of the predictor (default: 512)
        m: momentum of updating key encoder (default: 0.999)
        """
        super(BYOL, self).__init__()

        self.m = m

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
        self.model_momentum = copy.deepcopy(self.model)
        for p in self.model_momentum.parameters():
            p.requires_grad = False

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(feature_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(dim, feature_dim)) # output layer

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the target encoder
        """
        for param_q, param_k in zip(self.model.parameters(),
                                    self.model_momentum.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, t1, t2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2006.07733 for detailed notations
        """

        # compute features for one view
        h1 = self.model(x1) # NxC
        h2 = self.model(x2) # NxC

        p1 = self.predictor(h1) # NxC
        p2 = self.predictor(h2) # NxC

        with torch.no_grad():  # no gradient to targets
            self._momentum_update_target_encoder()  # update the key encoder

            t1 = self.model_momentum(x1)
            t2 = self.model_momentum(x2)

        return p1, p2, t1.detach(), t2.detach()