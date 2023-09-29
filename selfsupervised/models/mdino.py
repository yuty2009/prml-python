# Refer to: https://github.com/facebookresearch/dino/
#           https://github.com/KeremTurgutlu/self_supervised/
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys; sys.path.append(os.getcwd())
from common import distributed as dist
from common.head import MLPHead
from common.mask import MaskGenerator2d


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FlattenSeq(nn.Module):
    def forward(self, x):
        return x.reshape(-1, x.shape[-1]) # [N, L, C] -> [NL, C]

class MDINO(nn.Module):
    def __init__(
        self, encoder, encoder_dim=2048, feature_dim=256,
        n_mlplayers=3, hidden_dim=2048, bottleneck_dim=256,
        use_bn=False, norm_last_layer=True, momentum=0.999,
        image_size=224, patch_size=16,
    ):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output, your feature dimension (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 256)
        - n_mlplayers: number of MLP layers for the projector (default: 3)
        - hidden_dim: hidden dimension if a multi-layer projector was used
        - bottleneck_dim: bottleneck dimension if a multi-layer projector was used
        - use_bn: whether use batch normalization (default: False)
        - norm_last_layer: whether or not to weight normalize the last layer of the DINO head.
        - momentum: momentum of updating key encoder (default: 0.999)
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.
        (default: True)
        """
        super(MDINO, self).__init__()

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

        # create the online encoder
        self.encoder = encoder
        self.head_s = MLPHead(
            in_dim=encoder_dim, out_dim=feature_dim, n_layers=n_mlplayers, 
            hidden_dims=[hidden_dim, bottleneck_dim], activation=nn.GELU(), 
            use_bn=use_bn, norm_last_layer=norm_last_layer, 
        )
        self.head_t = MLPHead(
            in_dim=encoder_dim, out_dim=feature_dim, n_layers=n_mlplayers, 
            hidden_dims=[hidden_dim, bottleneck_dim], activation=nn.GELU(), 
            use_bn=use_bn, norm_last_layer=True, 
        )
        self.student = nn.Sequential(self.encoder, FlattenSeq(), self.head_s)
        self.teacher = nn.Sequential(copy.deepcopy(encoder), FlattenSeq(), self.head_t)
        # disable backpropagation through the teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.loss_fn = DINOLoss(out_dim=feature_dim)

    @torch.no_grad()
    def _momentum_update_teacher(self):
        for param_s, param_t in zip(self.student.parameters(),
                                    self.teacher.parameters()):
            param_t.data = param_t.data * self.momentum + param_s.data * (1. - self.momentum)

    def forward(self, x):
        mask = self.mask_generator(x.size(0))
        mask = torch.from_numpy(mask).flatten(1).to(x.device)
       
        x_patch = self.patchify(x)
        x_masked = x_patch * (1 - mask.unsqueeze(-1).float())
        x_masked = self.unpatchify(x_masked)

        output_s = self.student(x_masked)
        output_t = self.teacher(x)

        # EMA update for the teacher
        with torch.no_grad():
            self._momentum_update_teacher()
            
        # normalize feature embeddings (notice that it is already done in DINOHead)
        output_s = nn.functional.normalize(output_s, dim=1)
        output_t = nn.functional.normalize(output_t, dim=1)
        return self.loss_fn(output_s, output_t, mask), mask
    
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


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops=2, temperature_s=0.1, temperature_t=0.04, center_momentum=0.9):
        super().__init__()
        self.ncrops = ncrops
        self.temperature_s = temperature_s
        self.temperature_t = temperature_t
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, output_s, output_t, mask):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        out_s = output_s / self.temperature_s

        # teacher centering and sharpening
        out_t = F.softmax((output_t - self.center) / self.temperature_t, dim=-1)
        out_t = out_t.detach()

        self.update_center(output_t)

        loss = torch.sum(-out_t * F.log_softmax(out_s, dim=-1), dim=-1)
        loss = loss.reshape(mask.size(0), -1)
        loss_cls = loss[:, 0] # cls token
        loss_mask = loss[:, 1:] # remove the loss of cls token
        loss_mask = (loss_mask * mask).sum(dim=1) / mask.sum(dim=1)
        
        return (loss_mask + loss_cls).mean()

    @torch.no_grad()
    def update_center(self, output_t):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(output_t, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center /= len(output_t)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOHead(nn.Module):
    '''
    copy.deepcopy:
    RuntimeError: Only Tensors created explicitly by the user (graph leaves)
    support the deepcopy protocol at the moment
    https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
    https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    '''
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
