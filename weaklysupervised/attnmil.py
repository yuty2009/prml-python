
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    def __init__(self, encoder, num_classes=2, embed_dim=500, attention_dim=128):
        super(AttentionMIL, self).__init__()

        self.encoder = encoder
        encoder_dim = encoder.feature_dim

        self.feature_projector = nn.Sequential(
            nn.Linear(encoder_dim, embed_dim),
            nn.ReLU(),
        )

        self.attn_pooling = nn.Sequential(
            nn.Linear(embed_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """ 
        x: (B, N, C, H, W) batch_size x num_instances x channels x height x width
        """
        inshape = x.shape
        x = x.view(-1, *inshape[2:])

        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.feature_projector(x)  # B*N x P

        attn_w = self.attn_pooling(x)  # B*N x 1
        attn_w = attn_w.view(inshape[0], inshape[1])  # B x N
        attn_w = F.softmax(attn_w, dim=1)  # softmax over N

        x = x.view(inshape[0], inshape[1], -1)  # B x N x P
        x = torch.einsum('bn,bnp->bp', attn_w, x)  # B x P

        output = self.classifier(x)

        return output, attn_w
