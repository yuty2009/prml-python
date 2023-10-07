# refer to https://github.com/facebookresearch/mae
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import os, sys; sys.path.append(os.getcwd())
from common.mask import MaskGenerator2d
from common.modules import PatchEmbedding2d
from common.modules import SinCosPositionalEmbedding2d
from common.modules import TransformerEncoderLayer
from common.modules import TransformerEncoder


class MAE(nn.Module):
    def __init__(self, input_size=224, patch_size=16, in_chans=3, mask_prob=0.75,
                 embed_dim=1024, num_layers=24, num_heads=16, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, dropout_transformer=0.1,
                 embed_dim_decoder=512, num_layers_decoder=8,  num_heads_decoder=16,
                 norm_pix_loss=False):
        super().__init__()
        input_size = _pair(input_size)
        patch_size = _pair(patch_size)
        self.patch_size = patch_size
        self.num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])

        self.patch_embed = PatchEmbedding2d(patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.mask_generator = MaskGenerator2d(mask_prob=mask_prob, mask_type='random')
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = SinCosPositionalEmbedding2d(embed_dim, self.num_patches, True)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                embed_dim, num_heads, int(embed_dim*mlp_ratio), dropout=dropout_transformer
            ),
            num_layers,
        )
        self.norm_encoder = norm_layer(embed_dim)

        ## MAE decoder specifics
        self.patch_embed_decoder = nn.Linear(embed_dim, embed_dim_decoder, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim_decoder))
        # fix sin-cos positional encoding
        self.pos_embed_decoder = SinCosPositionalEmbedding2d(embed_dim_decoder, self.num_patches, True)
        # transformer blocks
        self.blocks_decoder = TransformerEncoder(
            TransformerEncoderLayer(
                embed_dim_decoder, num_heads_decoder, int(embed_dim_decoder*mlp_ratio),
            ),
            num_layers_decoder,
        )
        self.norm_decoder = norm_layer(embed_dim_decoder)

        # decoder to patch
        self.head_decoder = nn.Linear(embed_dim_decoder, self.patch_size[0]*self.patch_size[1]*in_chans)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        latent, mask = self.encode(x)
        pred = self.decode(latent, mask)  # [N, L, p*p*3]
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
            x, mask = self.apply_mask(x)
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (bs, num_patches+1, emb_dim)
        # add pos embed w/o cls token
        x = self.pos_embed(x)
        # apply Transformer blocks
        x = self.encoder(x)[0][-1] # only use the last layer
        x = self.norm_encoder(x)
        return x, mask

    def refill_mask(self, x, mask):
        ids_mask = mask.nonzero(as_tuple=False)
        ids_keep = (1-mask).nonzero(as_tuple=False)
        x_ = torch.zeros(mask.size(0), mask.size(1), x.size(-1), device=x.device)
        x_[ids_mask[:, 0], ids_mask[:, 1], :] = self.mask_token
        x_[ids_keep[:, 0], ids_keep[:, 1], :] = x[:, 1:, :].reshape(-1, x.size(-1))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        return x

    def decode(self, x, mask):
        # embed tokens
        x = self.patch_embed_decoder(x)
        # append mask tokens to sequence
        x = self.refill_mask(x, mask)
        # add pos embed
        x = self.pos_embed_decoder(x)

        # apply Transformer blocks
        x = self.blocks_decoder(x)[0][-1] # only use the last layer
        x = self.norm_decoder(x)

        # predictor projection
        x = self.head_decoder(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
    
    def loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / (mask.sum() + 1e-8)  # mean loss on removed patches
        return loss
    
    def forward_feature(self, x, pool='mean'):
        x = self.encode(x, need_mask=False)[0]
        if pool == 'mean':
            x = x[:, 1:, :].mean(dim=1) # pool without cls token
        else:
            x = x[:, 0] # cls token
        return x

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

    model = MAE(
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

    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt

    img = Image.open('/Users/yuty2009/data/prmldata/imagenet-1k/train/n01440764/n01440764_18.JPEG')
    img = img.resize((224, 224))
    img_data = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255
    x = model.patchify(img_data)
    # mask, ids_keep, ids_restore = model.generate_mask(img_data, 0.75)
    mask_generator = MaskGenerator2d(mask_prob=0.75, mask_type='random')
    mask = mask_generator(x.shape[:2])
    mask = torch.from_numpy(mask).flatten(1)
    mask_token_1 = torch.zeros(x.shape)
    w = mask.flatten(1).unsqueeze(-1).type_as(mask_token_1)
    x_masked_1 = x * (1 - w) + mask_token_1 * w
    # x_masked_2 = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(-1)))
    ids_mask = mask.nonzero(as_tuple=False)
    ids_keep = (1-mask).nonzero(as_tuple=False)
    x_masked_2 = x[ids_keep[:, 0], ids_keep[:, 1], :].reshape(x.size(0), -1, x.size(-1))
    # mask_tokens = torch.zeros(x_masked_2.size(0), ids_restore.size(1) - x_masked_2.size(1), x_masked_2.size(-1))
    x_rec = torch.zeros_like(x)
    x_rec[ids_mask[:, 0], ids_mask[:, 1], :] = 0
    x_rec[ids_keep[:, 0], ids_keep[:, 1], :] = x_masked_2
    # x_rec = torch.cat([x_masked_2, mask_tokens], dim=1)  # no cls token
    # x_rec = torch.gather(x_rec, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

    img_rec_1 = model.unpatchify(x_masked_1)
    img_rec_1 = img_rec_1[0].permute(1, 2, 0).numpy()
    img_rec_2 = model.unpatchify(x_rec)
    img_rec_2 = img_rec_2[0].permute(1, 2, 0).numpy()

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(img_rec_1)
    plt.subplot(1, 3, 3)
    plt.imshow(img_rec_2)
    plt.show()
    