
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnMIL(nn.Module):
    def __init__(self, feature_encoder, feature_dim=800, project_dim=500, attention_dim=128, num_classes=2):
        super(AttnMIL, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.project_dim = project_dim
        self.attention_dim = attention_dim

        self.feature_encoder = feature_encoder

        self.feature_projector = nn.Sequential(
            nn.Linear(feature_dim, self.project_dim),
            nn.ReLU(),
        )

        self.attn_pooling = nn.Sequential(
            nn.Linear(self.project_dim, self.attention_dim),
            nn.Tanh(),
            nn.Linear(self.attention_dim, 1)
        )

        self.classifier = nn.Linear(self.project_dim, num_classes)

    def forward(self, x):
        """ 
        x: (B, N, C, H, W) batch_size x num_instances x channels x height x width
        """
        inshape = x.shape
        x = x.view(-1, *inshape[2:])

        x = self.feature_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.feature_projector(x)  # B*N x P

        attn_w = self.attn_pooling(x)  # B*N x 1
        attn_w = attn_w.view(inshape[0], inshape[1])  # B x N
        attn_w = F.softmax(attn_w, dim=1)  # softmax over N

        x = x.view(inshape[0], inshape[1], -1)  # B x N x P
        x = torch.einsum('bn,bnp->bp', attn_w, x)  # B x P

        output = self.classifier(x)

        return output, attn_w
