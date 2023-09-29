
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import os, sys; sys.path.append(os.getcwd())
from common.modules import PatchEmbedding2d


class MLPMixerLayer(nn.Module):
    def __init__(self, num_patches, embed_dim, ratio_1=4, ratio_2=0.5, dropout=0.,):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        inner_dim_1 = int(num_patches*ratio_1)
        self.ff_1 = nn.Sequential(
            nn.Conv1d(num_patches, inner_dim_1, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(inner_dim_1, num_patches, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.ln_2 = nn.LayerNorm(embed_dim)
        inner_dim_2 = int(embed_dim*ratio_2)
        self.ff_2 = nn.Sequential(
            nn.Linear(embed_dim, inner_dim_2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim_2, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        x = x + self.ff_1(self.ln_1(x))
        x = x + self.ff_2(self.ln_2(x))
        return x
    

class MLPMixer(nn.Module):
    def __init__(self, num_classes=0, input_size=224, patch_size=16, in_chans=3, 
                 embed_dim=512, num_layers=6, mlp_ratio_1=4., mlp_ratio_2=.5,
                 norm_layer=nn.LayerNorm, droprate_mixer=0.1, droprate_embed=.0):
        super().__init__()
        input_size = _pair(input_size)
        patch_size = _pair(patch_size)
        self.patch_size = patch_size
        self.num_patches = (input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1])

        self.patch_embed = PatchEmbedding2d(patch_size=self.patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.drop_embed = nn.Dropout(droprate_embed)
        self.encoder = nn.Sequential(*[
            MLPMixerLayer(self.num_patches, embed_dim, mlp_ratio_1, mlp_ratio_2, dropout=droprate_mixer)
            for _ in range(num_layers)
        ])
        self.norm = norm_layer(embed_dim)
        # classfier head
        self.feature_dim = embed_dim
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.drop_embed(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
    

if __name__ == '__main__':

    model = MLPMixer(
        num_classes = 1000,
        input_size = (256, 256),
        patch_size = (16, 16),
        embed_dim = 768,
        num_layers = 12,
        mlp_ratio_1 = 4.0,
        mlp_ratio_2 = 0.5,
    )

    x = torch.randn((2, 3, 256, 256))
    output = model(x)
    print(output.shape)
