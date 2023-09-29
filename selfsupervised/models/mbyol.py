# Masked BYOL, refer to BYOL and iBOT
import copy
import torch
import torch.nn as nn
import os, sys; sys.path.append(os.getcwd())
from common.head import MLPHead
from common.mask import MaskGenerator2d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FlattenSeq(nn.Module):
    def forward(self, x):
        return x.reshape(-1, x.shape[-1]) # [N, L, C] -> [NL, C]

class MBYOL(nn.Module):
    def __init__(self, encoder, encoder_dim=2048, feature_dim=512, predict_dim=512,
        n_mlplayers=2, hidden_dim=2048, use_bn=True, momentum=0.999,
        image_size=224, patch_size=16,
    ):
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
        super(MBYOL, self).__init__()

        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0, \
               'Image dimensions must be divisible by the patch size.'
        num_patches = (ih // ph) * (iw // pw)
        self.num_patches = num_patches
        self.input_size = pair(image_size)
        self.patch_size = pair(patch_size)
        self.momentum = momentum

        self.mask_generator = MaskGenerator2d(mask_prob=0.65, mask_type='random')

        self.encoder = encoder
        self.head_s = MLPHead(
            in_dim=encoder_dim, out_dim=feature_dim, n_layers=n_mlplayers,
            hidden_dims=[hidden_dim]*(n_mlplayers-1), use_bn=use_bn,
        )
        self.head_t = MLPHead(
            in_dim=encoder_dim, out_dim=feature_dim, n_layers=n_mlplayers,
            hidden_dims=[hidden_dim]*(n_mlplayers-1), use_bn=use_bn,
        )
        
        self.student = nn.Sequential(self.encoder, FlattenSeq(), self.head_s)
        self.teacher = nn.Sequential(copy.deepcopy(encoder), FlattenSeq(), self.head_t)
        for p in self.teacher.parameters():
            p.requires_grad = False
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(feature_dim, predict_dim)) # output layer

    @torch.no_grad()
    def _momentum_update_teacher(self):
        """
        Momentum update of the target encoder
        """
        for param_q, param_k in zip(self.student.parameters(),
                                    self.teacher.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def forward(self, x):
        mask = self.mask_generator(x.size(0))
        mask = torch.from_numpy(mask).flatten(1).to(x.device)

        x_patch = self.patchify(x)
        x_masked = x_patch * (1 - mask.unsqueeze(-1).float())
        x_masked = self.unpatchify(x_masked)

        z1 = self.student(x_masked) # NLxC
        p1 = self.predictor(z1) # NLxC

        with torch.no_grad():  # no gradient to teacher network
            self._momentum_update_teacher()
            t1 = self.teacher(x)

        return self.loss(p1, t1.detach(), mask), mask
    
    def loss(self, p1, t1, mask):
        loss = -nn.CosineSimilarity(dim=1)(p1, t1)
        loss = loss.reshape(mask.size(0), -1)
        loss_cls = loss[:, 0] # cls token
        loss_mask = loss[:, 1:] # remove the loss of cls token
        loss_mask = (loss_mask * mask).sum(dim=1) / mask.sum(dim=1)
        return (loss_mask + loss_cls).mean()
    
    def forward_feature(self, x, pool='mean'):
        x = self.encoder(x)
        if pool == 'mean':
            x = x[:, 1:, :].mean(dim=1) # pool without cls token
        else:
            x = x[:, 0] # cls token
        return x.squeeze()
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = x.permute(0, 2, 4, 3, 5, 1)
        # x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = x.permute(0, 5, 1, 3, 2, 4)
        # x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    

if __name__ == '__main__':

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from vit_timm import ViT

    encoder = ViT(
        image_size = (256, 256),
        patch_size = (16, 16),
        num_classes = 0,
        embed_dim = 768,
        num_layers = 12,
        num_heads = 12,
        mlp_ratio = 4.0,
        pool = 'none',
    )
    model = MBYOL(encoder, encoder_dim=768, image_size=256, patch_size=16)

    x = torch.randn((2, 3, 256, 256))
    loss = model(x)
    print(loss.shape)
