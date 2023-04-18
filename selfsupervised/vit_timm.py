
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

        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0, \
               'Image dimensions must be divisible by the patch size.'
        num_patches = (ih // ph) * (iw // pw)

        self.pool = pool
        assert pool in {'cls', 'mean'}, \
               'pool type must be either cls (cls token) or mean (mean pooling).'

        # patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(ph, pw), stride=(ph, pw)
        )
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
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, img):
        x = self.patch_embed(img) # (bs, emb_dim, n_ph, n_pw)
        x = x.flatten(2).permute(0, 2, 1) # (bs, num_patches, emb_dim)
        # prepend class token
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], dim=1) # (bs, num_patches+1, emb_dim)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        if self.pool == 'mean':
            x = x[:, 1:, :].mean(dim=1) # pool without cls token
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0] # cls token
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