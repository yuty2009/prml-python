# Copy from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
import copy
import torch
import torch.nn as nn


class BYOL(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, m=0.999):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        m: momentum of updating key encoder (default: 0.999)
        """
        super(BYOL, self).__init__()

        self.m = m

        # create the online encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_online = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder_online.fc.weight.shape[1]
        self.encoder_online.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder_online.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder_online.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # create the target encoder
        self.encoder_target = copy.deepcopy(self.encoder_online)
        for p in self.encoder_target.parameters():
            p.requires_grad = False

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        """
        Momentum update of the target encoder
        """
        for param_q, param_k in zip(self.encoder_online.parameters(),
                                    self.encoder_target.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2006.07733 for detailed notations
        """

        # compute features for one view
        h1 = self.encoder_online(x1) # NxC
        h2 = self.encoder_online(x2) # NxC

        p1 = self.predictor(h1) # NxC
        p2 = self.predictor(h2) # NxC

        with torch.no_grad():  # no gradient to targets
            self._momentum_update_target_encoder()  # update the key encoder

            z1 = self.encoder_target(x1)
            z2 = self.encoder_target(x2)

        return p1, p2, z1.detach(), z2.detach()