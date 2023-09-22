
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import SinCosPositionalEmbedding1d
from modules import TransformerEncoderLayer
from modules import TransformerEncoder


class TransformerMIL(nn.Module):
    def __init__(self, encoder, num_classes=2,
                 embed_dim=192, num_heads=6, num_layers=1, mlp_ratio=4,
                 norm_layer=nn.LayerNorm, dropout_transformer=0.1):
        super(TransformerMIL, self).__init__()

        self.encoder = encoder
        encoder_dim = encoder.feature_dim
        num_patches = encoder.num_patches

        self.feature_projector = nn.Sequential(
            nn.Linear(encoder_dim, embed_dim),
            nn.ReLU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = SinCosPositionalEmbedding1d(embed_dim, num_patches, True)
        self.seq_encoder = TransformerEncoder(
            TransformerEncoderLayer(
                embed_dim, num_heads, int(embed_dim*mlp_ratio), dropout=dropout_transformer
            ),
            num_layers,
        )
        self.norm = norm_layer(embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)

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
        """ 
        x: (B, N, C, H, W) batch_size x num_instances x channels x height x width
        """
        inshape = x.shape
        x = x.view(-1, *inshape[2:])

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.feature_projector(x)  # B*N x P
        x = x.view(inshape[0], inshape[1], -1)  # B x N x P

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (bs, num_patches+1, emb_dim)
        # add pos embed w/o cls token
        x = self.pos_embed(x)
        # apply Transformer blocks
        xs, attn_ws = self.seq_encoder(x)

        x = xs[-1] # last layer output
        attn_w = attn_ws[-1] # last layer attention weights
        x = self.norm(x)
        x = x[:, 0] # pooling by cls token

        output = self.classifier(x)

        return output, attn_w
