
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import os, sys; sys.path.append(os.getcwd())
from common.modules import PatchEmbedding2d
from common.modules import TransformerCausalLayer


class CGPT2Model(nn.Module):
    """ Continuous GPT2 Model for real-valued data """
    def __init__(
            self, max_seqlen=512, patch_size=16, in_chans=3, embed_drop=0.1,
            embed_dim=192, num_layers=6, num_heads=6, mlp_ratio=4., attn_drop=0.1,
        ):
        super().__init__()
        self.max_seqlen = max_seqlen
        self.patch_embed = PatchEmbedding2d(patch_size, in_chans, embed_dim)
        # start of sequence token
        self.sos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.sos_token)
        self.pos_embed = nn.Embedding(self.max_seqlen, embed_dim)
        self.drop = nn.Dropout(embed_drop)
        self.ln_f = nn.LayerNorm(embed_dim)

        self.blocks = nn.ModuleList([
            TransformerCausalLayer(
                d_model=embed_dim,
                n_heads=num_heads,
                d_ff=int(mlp_ratio*embed_dim),
                dropout=attn_drop
            ) for _ in range(num_layers)
        ])

        # init all weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        x = self.patch_embed(x) # token embeddings of shape (b, t, n_embd)
        
        T = x.size(1)
        assert T < self.max_seqlen, \
            f"Cannot forward sequence of length {T}, block size is only {self.max_seqlen}"
        pos = torch.arange(0, T+1, dtype=torch.long, device=x.device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        sos_token = self.sos_token.expand(x.size(0), -1, -1)
        x = torch.cat((sos_token, x), dim=1) # (bs, num_patches+1, emb_dim)
        pos_emb = self.pos_embed(pos) # position embeddings of shape (1, t, n_embd)
        x = self.drop(x + pos_emb[:, :T+1, :])
        for block in self.blocks:
            x, w = block(x)
        x = self.ln_f(x)
        return x

    
class CGPTARModel(nn.Module):
    """ GPT Language Model """
    def __init__(
            self, num_classes=0, input_size=224, patch_size=16, in_chans=3, embed_drop=0.1,
            embed_dim=192, num_layers=6, num_heads=6, mlp_ratio=4., attn_drop=0.1,
        ):
        super().__init__()
        input_size = _pair(input_size)
        patch_size = _pair(patch_size)
        self.patch_size = patch_size
        self.num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])
        self.transformer = CGPT2Model(
            self.num_patches+1, patch_size, in_chans, embed_drop,
            embed_dim, num_layers, num_heads, mlp_ratio, attn_drop,
        )
        self.lm_head = nn.Linear(embed_dim, self.patch_size[0]*self.patch_size[1]*in_chans)
        self.clf_head = nn.Linear(embed_dim, num_classes)

    def forward(self, inputs, labels=None):
        x_patches = self.patchify(inputs) # token embeddings of shape (b, t, n_embd)
        x = self.transformer(inputs)
        if labels is None: # self-supervised training
            x = x[:, :-1, :] # skip the last token
            logits = self.lm_head(x)
            loss = F.mse_loss(logits.reshape(-1, logits.size(-1)), x_patches.reshape(-1, x_patches.size(-1)))
        else:
            x = x[:, -1, :] # only take the last token
            logits = self.clf_head(x)
            loss = F.cross_entropy(logits, labels)
        return loss, logits
    
    def forward_feature(self, inputs):
        x = self.transformer(inputs)
        x = x[:, -1, :] # only take the last token
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

    model = CGPTARModel(
        num_classes=10, input_size=224, patch_size=16, in_chans=3, embed_drop=0.1,
        embed_dim=384, num_layers=6, num_heads=6
    )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M parameters")
    x = torch.randn((2, 3, 224, 224))
    _, output = model(x)
    print(output.shape)
