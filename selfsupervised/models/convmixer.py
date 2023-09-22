# https://openreview.net/forum?id=TVHS5Y4dNvM

import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, num_classes=0, patch_size=16, in_chans=3, 
                 embed_dim=512, num_layers=6, kernel_size=9,
                 droprate_mixer=0.1, droprate_embed=.0):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim),
        )
        self.drop_embed = nn.Dropout(droprate_embed)
        self.encoder = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, kernel_size, groups=embed_dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(embed_dim)
                )),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(embed_dim)
            ) for _ in range(num_layers)],
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )
        # classfier head
        self.feature_dim = embed_dim
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.drop_embed(x)
        x = self.encoder(x)
        return self.head(x)
    

if __name__ == '__main__':

    model = ConvMixer(
        num_classes = 1000,
        patch_size = (16, 16),
        embed_dim = 768,
        num_layers = 12,
    )

    x = torch.randn((2, 3, 256, 256))
    output = model(x)
    print(output.shape)
