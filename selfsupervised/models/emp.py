# https://github.com/tsb0601/EMP-SSL
import torch
import torch.nn as nn
import torch.nn.functional as F


class EMP(nn.Module):
    def __init__(self, encoder, encoder_dim=2048, feature_dim=512, hidden_dim=2048):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 512)
        - hidden_dim: hidden dimension if a multi-layer projector was used (default: 2048)
        """
        super().__init__()

        # create the online encoder
        self.encoder = encoder
        self.pre_feature = nn.Sequential(
            nn.Linear(encoder_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = torch.cat(inputs, dim=0) 
        feature = self.encoder(inputs)
        feature = self.pre_feature(feature)
        z = F.normalize(self.projector(feature), p=2)
        return z
        

class EMPLoss(nn.Module):
    def __init__(self, num_views=2, alpha=200):
        super().__init__()
        self.alpha = alpha
        self.num_views = num_views
        self.tcr = TotalCodingRate(eps=0.2)

    def forward(self, output):
        z_avg = output.mean(dim=0)
        z_list = output.chunk(self.num_views)
        # invariance Loss
        z_sim = 0
        for i in range(self.num_views):
            z_sim += F.cosine_similarity(z_list[i], z_avg, dim=1).mean()
        loss_inv = z_sim/self.num_views
        # TCR Loss
        loss_tcr = 0
        for i in range(self.num_views):
            loss_tcr += self.tcr(z_list[i])
        loss_tcr = loss_tcr/self.num_views
        # Total Loss
        loss = self.alpha * loss_inv + loss_tcr
        return loss

    
class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)
    