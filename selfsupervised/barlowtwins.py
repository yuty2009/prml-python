# Refer to: https://github.com/facebookresearch/barlowtwins/
import torch
import torch.nn as nn
from head import MLPHead


class BarlowTwins(nn.Module):
    def __init__(self, encoder, encoder_dim=2048, feature_dim=512, 
        n_mlplayers=2, hidden_dim=2048, use_bn=False):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output, your feature dimension (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 512)
        - n_mlplayers: number of MLP layers for the projector (default: 2)
        - hidden_dim: hidden dimension if a multi-layer projector was used (default: 2048)
        - use_bn: whether use batch normalization (default: False)
        """
        super(BarlowTwins, self).__init__()

        # create the online encoder
        self.encoder = encoder
        # create the online projector
        self.projector = MLPHead(
            in_dim=encoder_dim, out_dim=feature_dim, n_layers=n_mlplayers,
            hidden_dims=[hidden_dim]*(n_mlplayers-1), use_bn=use_bn,
        )

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        return z1, z2


class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        
    def forward(self, z1, z2):
        # empirical cross-correlation matrix
        c = torch.mm(z1.T, z2)/z1.size(0)
        # sum the cross-correlation matrix between all gpus
        torch.distributed.all_reduce(c)
        # loss
        c_diff = (c - torch.eye(z1.size(1), device=c.device)).pow(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(z1.size(1), dtype=bool)] *= self.lambda_param
        loss = c_diff.sum()
        return loss