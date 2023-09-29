# refer to https://github.com/microsoft/SimMIM/blob/main/models/simmim.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys; sys.path.append(os.getcwd())
from common.mask import MaskGenerator2d
from common.modules import PatchEmbedding2d
from common.modules import SinCosPositionalEmbedding2d
from common.modules import TransformerEncoderLayer
from common.modules import TransformerEncoder


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SimMIM(nn.Module):
    def __init__(self, input_size=224, patch_size=16, in_chans=3, mask_prob=0.75,
                 embed_dim=1024, num_layers=24, num_heads=16, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, dropout_transformer=0.1,
            ):
        super().__init__()
        input_size = pair(input_size)
        patch_size = pair(patch_size)
        self.patch_size = patch_size
        self.num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])

        self.patch_embed = PatchEmbedding2d(patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.mask_generator = MaskGenerator2d(mask_prob=mask_prob, mask_type='random')
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) # mask token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = SinCosPositionalEmbedding2d(embed_dim, self.num_patches, True)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                embed_dim, num_heads, int(embed_dim*mlp_ratio), dropout=dropout_transformer
            ),
            num_layers,
        )
        self.norm_encoder = norm_layer(embed_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels = embed_dim,
                out_channels = self.patch_size[0] * self.patch_size[1] * 3, kernel_size=1),
            nn.PixelShuffle(self.patch_size[0]),
        )

    def forward(self, x):
        latent, mask = self.encode(x)
        latent = latent[:, 1:, :]  # remove cls token
        pred = self.decode(latent)  # [N, L, p*p*3]
        loss = self.loss(x, pred, mask)
        return loss, pred, mask
    
    def apply_mask(self, x, keep_mask=False):
        mask = self.mask_generator((x.size(0), x.size(1)))
        mask = torch.from_numpy(mask).to(x.device)
        mask = mask.flatten(1)
        if not keep_mask:
            ids_keep = (1-mask).nonzero(as_tuple=False)
            x = x[ids_keep[:, 0], ids_keep[:, 1], :].reshape(x.size(0), -1, x.size(-1))
        else:
            mask_token = self.mask_token.expand(x.size(0), x.size(1), -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w
        return x, mask
    
    def encode(self, x, need_mask=True):
        # embed patches
        x = self.patch_embed(x)
        mask = None
        if need_mask:
            # apply mask
            x, mask = self.apply_mask(x, keep_mask=True)
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (bs, num_patches+1, emb_dim)
        # add pos embed w/o cls token
        x = self.pos_embed(x)
        # apply Transformer blocks
        x = self.encoder(x)[0][-1] # only use the last layer
        x = self.norm_encoder(x)
        return x, mask
    
    def decode(self, latent):
        B, L, D = latent.shape
        H = W = int(L ** 0.5)
        latent = latent.permute(0, 2, 1).reshape(B, D, H, W)
        pred = self.decoder(latent)
        pred = pred.permute(0, 2, 3, 1).reshape(B, H*W, -1)
        return pred
    
    def loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        loss_recon = F.l1_loss(target, pred, reduction='none')
        loss_recon = loss_recon.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-8)
        return loss
    
    def forward_feature(self, x, pool='mean'):
        x = self.encode(x, need_mask=False)[0]
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

    from vit_timm import ViT

    model = SimMIM(
        input_size = (224, 224),
        patch_size = (16, 16),
        in_chans = 3,
        mask_prob = 0.75,
        embed_dim = 768,
        num_layers = 12,
        num_heads = 12,
        mlp_ratio = 4.0,
    )

    x = torch.randn((2, 3, 224, 224))
    loss, x_rec, mask = model(x)
    print(x_rec.shape)
    print(mask.shape)
    