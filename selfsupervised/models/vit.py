
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import os, sys; sys.path.append(os.getcwd())
from common.modules import PatchEmbedding2d
from common.modules import SinCosPositionalEmbedding2d
from common.modules import TransformerEncoderLayer
from common.modules import TransformerEncoder


class ViT(nn.Module):
    def __init__(self, num_classes=0, input_size=224, patch_size=16, in_chans=3, 
                 embed_dim=1024, num_layers=24, num_heads=16, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, droprate_trans=0.1,
                 droprate_embed=.0, pool = 'cls'):
        super().__init__()
        self.pool = pool
        input_size = _pair(input_size)
        patch_size = _pair(patch_size)
        self.patch_size = patch_size
        self.num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])

        self.patch_embed = PatchEmbedding2d(patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = SinCosPositionalEmbedding2d(embed_dim, self.num_patches, True)
        self.drop_embed = nn.Dropout(droprate_embed)
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                embed_dim, num_heads, int(embed_dim*mlp_ratio), dropout=droprate_trans
            ),
            num_layers,
        )
        self.norm = norm_layer(embed_dim) if pool != 'mean' else nn.Identity()
        # classfier head
        self.feature_dim = embed_dim
        self.fc_norm = norm_layer(embed_dim) if pool == 'mean' else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

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

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x):
        x = self.forward_feature(x)
        if self.pool == 'mean':
            x = x[:, 1:, :].mean(dim=1) # pool without cls token
        elif self.pool == 'cls':
            x = x[:, 0] # return the cls token
        x = self.fc_norm(x)
        return self.head(x)

    def forward_feature(self, x):
        # embed patches
        x = self.patch_embed(x)
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (bs, num_patches+1, emb_dim)
        # add pos embed w/o cls token
        x = self.pos_embed(x)
        x = self.drop_embed(x)
        # apply Transformer blocks
        x = self.encoder(x)[0][-1] # only use the last layer
        x = self.norm(x)
        return x
    

if __name__ == '__main__':

    model = ViT(
        num_classes = 1000,
        input_size = (256, 256),
        patch_size = (16, 16),
        embed_dim = 768,
        num_layers = 12,
        num_heads = 12,
        mlp_ratio = 4.0,
    )

    x = torch.randn((2, 3, 256, 256))
    output = model(x)
    print(output.shape)
