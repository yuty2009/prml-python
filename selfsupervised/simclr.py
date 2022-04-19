# Refer to: https://github.com/google-research/simclr/blob/master/model.py
#       and https://github.com/Spijkervet/SimCLR/simclr/modules/nt_xent.py
#       and https://github.com/dtheo91/simclr/blob/master/modules/utils/loss_functions.py
#       and https://github.com/KeremTurgutlu/self_supervised/self_supervised/vision/simclr.py
import torch
import torch.nn as nn

from gather import GatherLayer


class SimCLR(nn.Module):
    def __init__(self, encoder, encoder_dim=2048, feature_dim=512, dim=128, num_mlplayers=2):
        """
        encoder: encoder you want to use to get feature representations (eg. resnet50)
        encoder_dim: dimension of the encoder output, your feature dimension (default: 2048 for resnets)
        feature_dim: intermediate dimension of the projector (default: 512)
        dim: projection dimension (default: 128)
        """
        super(SimCLR, self).__init__()

        self.encoder = encoder
        if num_mlplayers == 2:
            self.projector = nn.Sequential(nn.Linear(encoder_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(feature_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        elif num_mlplayers == 3:
            self.projector = nn.Sequential(nn.Linear(encoder_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(feature_dim, dim, bias=False),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        return z1, z2


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()

    def forward(self, z1, z2):
        N = z1.shape[0] # * self.world_size
        z = torch.cat((z1, z2), dim=0)  # [2N, D]
        # z = torch.cat(GatherLayer.apply(z), dim=0)
        # [2N, 2N]
        sim = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
        idx = torch.arange(2*N).roll(N)
        labels = torch.eye(2*N, device=sim.device)[idx]

        mask = torch.ones((2*N, 2*N), dtype=bool).fill_diagonal_(0)
        # [2N, 2N-1]
        logits = sim[mask].reshape(2*N, -1)
        labels = labels[mask].reshape(2*N, -1).nonzero(as_tuple=False)[:,-1]
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


class NTXentLoss1(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size() 

    def forward(self, z1, z2):
        N = z1.shape[0] # * self.world_size
        z = torch.cat((z1, z2), dim=0)  # [2N, D]
        # z = torch.cat(GatherLayer.apply(z), dim=0)

        # sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)

        sim_12 = torch.diag(sim, N)
        sim_21 = torch.diag(sim, -N)

        mask = torch.ones((2*N, 2*N), dtype=bool).fill_diagonal_(0)
        for i in range(N):
            mask[i, N + i] = 0
            mask[N + i, i] = 0

        positive_samples = torch.cat((sim_12, sim_21), dim=0).reshape(2*N, 1)
        negative_samples = sim[mask].reshape(2*N, -1)

        labels = torch.zeros(2*N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = nn.functional.cross_entropy(logits, labels)
        return loss


class NTXentLoss2(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        N = z1.shape[0]
        # [2*B, D]
        out = torch.cat([z1, z2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * N, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * N, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss


if __name__ == '__main__':

    import torchvision.models as models
    model = SimCLR(models.__dict__['resnet50'])
    loss_fn = NTXentLoss()

    z1, z2 = torch.rand([4, 5]), torch.rand([4, 5])
    loss = loss_fn(z1, z2)
