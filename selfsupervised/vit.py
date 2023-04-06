"""
References:
    https://github.com/lucidrains/vit-pytorch
    https://github.com/asyml/vision-transformer-pytorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer copy-paste from torch.nn.TransformerEncoderLayer
        with modifications:
        * layer norm before add residual
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False, normalize_before=True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward_post(self, src, src_mask = None, src_key_padding_mask = None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask = None, src_key_padding_mask = None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask)
        return self.forward_post(src, src_mask, src_key_padding_mask)


class Transformer(nn.Module):
    """Transformer class copy-paste from torch.nn.Transformer
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", layer_norm_eps: float = 1e-5, 
                 batch_first: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Transformer, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, batch_first,
                                                **factory_kwargs)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self._reset_parameters()

    def forward(self, src):
        output = self.encoder(src)
        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class ViT(nn.Module):
    def __init__(self, num_classes, image_size=224, patch_size=16,
                 d_model = 768, nhead = 12, num_layers = 12, ffn_dim = 3072,
                 pool = 'cls', in_channels = 3, dropout = 0.1):
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
        self.patch_embedding = nn.Conv2d(
            in_channels, d_model, kernel_size=(ph, pw), stride=(ph, pw)
        )
        # class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.pos_drop = nn.Dropout(dropout)
        # transformer
        self.transformer = Transformer(
            d_model = d_model,
            nhead = nhead,
            num_layers = num_layers,
            dim_feedforward = 3 * d_model,
            batch_first = True,
        )
        # classfier
        # self.fc = nn.Linear(emb_dim, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, num_classes),
            nn.Dropout(dropout)
        )

    def forward(self, img):
        x = self.patch_embedding(img) # (bs, emb_dim, n_ph, n_pw)
        x = x.flatten(2).permute(0, 2, 1) # (bs, num_patches, emb_dim)
        # prepend class token
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat([cls_token, x], dim=1) # (bs, num_patches+1, emb_dim)
        x = self.pos_drop(x + self.pos_embedding)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        out = self.fc(x)
        return out


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == '__main__':

    model = ViT(
        image_size=(256, 256),
        patch_size=(16, 16),
        num_classes=1000,
        d_model = 768,
        nhead = 12,
        num_layers = 12,
        ffn_dim = 3072,
    )

    x = torch.randn((2, 3, 256, 256))
    output = model(x)
    print(output.shape)
