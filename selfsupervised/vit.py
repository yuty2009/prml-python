"""
References:
    https://github.com/lucidrains/vit-pytorch
    https://github.com/asyml/vision-transformer-pytorch
"""
import torch
import torch.nn as nn
from common.attention import MultiheadAttention


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer copy-paste from torch.nn.TransformerEncoderLayer
        with modifications:
        * layer norm before add residual
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop_rate=0., attn_drop_rate=0.,
                 activation=nn.GELU, norm_layer=nn.LayerNorm, normalize_before=True):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=attn_drop_rate)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(embed_dim, int(mlp_ratio*embed_dim))
        self.dropout = nn.Dropout(drop_rate)
        self.linear2 = nn.Linear(int(mlp_ratio*embed_dim), embed_dim)

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)

        self.activation = activation()
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask = None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask = None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask)
        return self.forward_post(src, src_mask)


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
            TransformerEncoderLayer(
                embed_dim = embed_dim, num_heads = num_heads, mlp_ratio = mlp_ratio,
                drop_rate = drop_rate, attn_drop_rate = attn_drop_rate,
                activation = nn.GELU, norm_layer = nn.LayerNorm, normalize_before = True)
            for _ in range(num_layers)])
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


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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
