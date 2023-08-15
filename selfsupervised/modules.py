# copy from https://github.com/bshall/hubert/
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import DropPath
    

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=384, norm_layer=None, flatten=True):
        super().__init__()
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2) # (B, C, N)
        x = x.transpose(-2, -1)  # BCN -> BNC and normlize on the feature dimention
        x = self.norm(x)
        return x
    

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)
    

class FeatureProjection(nn.Module):
    def __init__(self, in_features=512, out_features=768, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.projection = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x
    

class SinCosPositionalEmbedding1d(nn.Module):
    """
    embed_dim: output dimension for each position
    max_len: maximum position index
    """
    def __init__(self, embed_dim, seq_len, cls_token=False):
        super().__init__()
    
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = torch.arange(seq_len, dtype=float)  # (M,)
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out) # (M, D/2)
        emb_cos = torch.cos(out) # (M, D/2)

        self.posembed = torch.concat([emb_sin, emb_cos], axis=1)  # (M, D)

        if cls_token:
            self.posembed = torch.cat([torch.zeros([1, embed_dim]), self.posembed], axis=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.posembed[:x.size(1)].to(x.dtype).to(x.device)
        return x
    

class SinCosPositionalEmbedding2d(nn.Module):
    """
    :param embed_dim: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    def __init__(self, embed_dim, seq_len, cls_token=False):
        super().__init__()

        height = width = int(math.sqrt(seq_len))
        assert(height * width == seq_len), "seq_len should be a square number"
    
        if embed_dim % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(embed_dim))
        pe = torch.zeros(height, width, embed_dim)
        # Each dimension use half of d_model
        embed_dim = int(embed_dim / 2)
        div_term = torch.exp(torch.arange(0., embed_dim, 2) *
                            -(math.log(10000.0) / embed_dim))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[:, :, 0:embed_dim:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
        pe[:, :, 1:embed_dim:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(height, 1, 1)
        pe[:, :,  embed_dim::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)
        pe[:, :, embed_dim + 1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1, width, 1)

        self.posembed = pe.reshape(-1, embed_dim*2)

        if cls_token:
            self.posembed = torch.cat([torch.zeros([1, embed_dim*2]), self.posembed], axis=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.posembed[:x.size(1)].to(x.dtype).to(x.device)
        return x
    

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.attn_ln = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ff_ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 1. self attention
        x = self.attn_ln(x)
        attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=mask)
        x = x + attn_out
        # 2. ff network
        x = x + self.ff(self.ff_ln(x))
        return x, attn_weights.detach()
    
    
class TransformerEncoderLayerTimm(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 drop_path: float = 0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # 1. self attention
        x = self.norm1(x)
        attn_out, attn_weights = self.attn(x, x, x, key_padding_mask=mask)
        x = x + self.drop_path1(attn_out)
        # 2. ff network
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x, attn_weights.detach()
    

class TransformerEncoder(nn.Module):
    def __init__(
        self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        layer_outputs = []
        attn_weights = []
        for layer in self.layers:
            x, w = layer(x, mask=mask)
            layer_outputs.append(x)
            attn_weights.append(w)

        return layer_outputs, attn_weights
    

if __name__ == '__main__':

    posembed = SinCosPositionalEmbedding1d(128, 100)
    
    from matplotlib import pyplot as plt

    plt.figure()
    plt.imshow(posembed.posembed.detach().cpu().numpy(), aspect='auto')
    plt.show()
