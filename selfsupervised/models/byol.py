# Copy from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
import copy
import torch
import torch.nn as nn
import os, sys; sys.path.append(os.getcwd())
from common.head import MLPHead


class BYOL(nn.Module):
    """
    Build a BYOL model. This model is adapted from SimSiam
    """
    def __init__(self, encoder, encoder_dim=2048, feature_dim=512, predict_dim=512,
        n_mlplayers=2, hidden_dim=2048, use_bn=True,
        momentum=0.999):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output, your feature dimension (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 512)
        - predict_dim: hidden dimension of the predictor (default: 512)
        - n_mlplayers: number of MLP layers for the projector (default: 2)
        - hidden_dim: hidden dimension if a multi-layer projector was used (default: 2048)
        - use_bn: whether use batch normalization (default: True)
        - momentum: momentum of updating key encoder (default: 0.999)
        """
        super(BYOL, self).__init__()

        self.momentum = momentum

        # create the online encoder
        self.encoder = encoder
        # create the online projector
        self.projector = MLPHead(
            in_dim=encoder_dim, out_dim=feature_dim, n_layers=n_mlplayers,
            hidden_dims=[hidden_dim]*(n_mlplayers-1), use_bn=use_bn,
        )
        
        self.model = nn.Sequential(self.encoder, self.projector)
        self.model_momentum = copy.deepcopy(self.model)
        for p in self.model_momentum.parameters():
            p.requires_grad = False
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(feature_dim, predict_dim)) # output layer

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the target encoder
        """
        for param_q, param_k in zip(self.model.parameters(),
                                    self.model_momentum.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

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