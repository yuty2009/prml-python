# Refer to: https://github.com/google-research/simclr/blob/master/model.py
#       and https://github.com/Spijkervet/SimCLR/simclr/modules/nt_xent.py
#       and https://github.com/dtheo91/simclr/blob/master/modules/utils/loss_functions.py
#       and https://github.com/KeremTurgutlu/self_supervised/self_supervised/vision/simclr.py
import torch
import torch.nn as nn
from .head import MLPHead


class SimCLR(nn.Module):
    def __init__(self, encoder, encoder_dim=2048, feature_dim=512,
        n_mlplayers=2, hidden_dim=2048, use_bn=False):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 512)
        - n_mlplayers: number of MLP layers for the projector (default: 2)
        - hidden_dim: hidden dimension if a multi-layer projector was used (default: 2048)
        - use_bn: whether use batch normalization (default: False)
        """
        super(SimCLR, self).__init__()

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


if __name__ == '__main__':

    import torchvision.models as models
    from selfsupervised.models.ntxent import NTXentLoss

    model = SimCLR(models.__dict__['resnet50'])
    loss_fn = NTXentLoss()

    z1, z2 = torch.rand([4, 5]), torch.rand([4, 5])
    loss = loss_fn(z1, z2)
