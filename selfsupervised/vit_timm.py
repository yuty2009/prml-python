
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_chans=3, num_classes=0, 
                 embed_dim=768, num_layers=12, num_heads=12, mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 pool = 'cls'):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0, \
               'Image dimensions must be divisible by the patch size.'
        num_patches = (ih // ph) * (iw // pw)
        self.num_patches = num_patches
        self.patch_size = pair(patch_size)
        self.pool = pool

        # patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(ph, pw), stride=(ph, pw)
        )
        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        # transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(num_layers)])
        self.norm = norm_layer(embed_dim)
        # classfier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

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

    def forward(self, img, mask=None, keep_mask=False):
        # img: (bs, c, h, w)
        # mask: (bs, len_keep) shuffled mask ids or (bs, num_patches) mask
        # keep_mask: whether to keep the masked patches in the output
        x = self.patch_embed(img) # (bs, emb_dim, n_ph, n_pw)
        x = x.flatten(2).permute(0, 2, 1) # (bs, num_patches, emb_dim)
        # mask patches
        if mask is not None:
            if not keep_mask:
                x = torch.gather(x, dim=1, index=mask.unsqueeze(-1).repeat(1, 1, x.size(-1)))
            else:
                mask_token = self.mask_token.expand(x.size(0), x.size(1), -1)
                w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
                x = x * (1 - w) + mask_token * w
        # prepend class token
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], dim=1) # (bs, num_patches+1, emb_dim)
        x = self.pos_drop(x + self.pos_embed[:, :(x.size(1)), :])
        for blk in self.blocks:
            x = blk(x)
        if self.pool == 'mean':
            x = x[:, 1:, :].mean(dim=1) # pool without cls token
            x = self.norm(x)
        elif self.pool == 'cls':
            x = self.norm(x)
            x = x[:, 0] # cls token
        else: # return all tokens
            x = x[:, 1:, :]
        out = self.head(x)
        return out
    

if __name__ == '__main__':

    model = ViT(
        image_size = (256, 256),
        patch_size = (16, 16),
        num_classes = 1000,
        embed_dim = 768,
        num_layers = 12,
        num_heads = 12,
        mlp_ratio = 4.0,
    )

    x = torch.randn((2, 3, 256, 256))
    output = model(x)
    print(output.shape)