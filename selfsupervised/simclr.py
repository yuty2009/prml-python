# Refer to: https://github.com/google-research/simclr/blob/master/model.py
#       and https://github.com/Spijkervet/SimCLR/simclr/modules/nt_xent.py
#       and https://github.com/dtheo91/simclr/blob/master/modules/utils/loss_functions.py
#       and https://github.com/KeremTurgutlu/self_supervised/self_supervised/vision/simclr.py
import torch
import torch.nn as nn

from gather import GatherLayer


class SimCLR(nn.Module):
    def __init__(self, encoder, n_features=2048, dim=128, T=0.07, proj_bn=False, world_size=1):
        """
        encoder: Encoder you want to use to get feature representations (eg. resnet18)
        n_features: The dimension of the encoder output, your feature dimension
        dim: Projection dimension (default: 128)
        T: Softmax temperature (default: 0.07)
        proj_bn: Applying batch normalization or not in projector
        """
        super(SimCLR, self).__init__()

        self.T = T
        self.world_size = world_size
        self.n_features = n_features
        self.encoder = encoder
        self.similarity_f = nn.CosineSimilarity(dim=2)

        if proj_bn:
            self.projector = nn.Sequential (
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.BatchNorm1d(num_features=self.n_features),
                nn.ReLU(),
                nn.Linear(self.n_features, dim, bias=False),
                nn.BatchNorm1d(num_features=dim)
            )
        else:
            self.projector = nn.Sequential (
                nn.Linear(self.n_features, self.n_features, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_features, dim, bias=False),
            )

    def forward(self, x_i, x_j):
        # encoding
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        # projection
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        # normalize feature embeddings
        # z_i = nn.functional.normalize(z_i, p=2, dim=1)
        # z_j = nn.functional.normalize(z_j, p=2, dim=1)

        logits, labels = self.calc_loss_1(z_i, z_j)

        return logits, labels

    def calc_loss_1(self, z_i, z_j):
        N = z_i.shape[0] * self.world_size
        z = torch.cat((z_i, z_j), dim=0)  # [2N, D]
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)
        # [2N, 2N]
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.T
        idx = torch.arange(2*N).roll(N)
        labels = torch.eye(2*N, device=sim.device)[idx]

        mask = torch.ones((2*N, 2*N), dtype=bool).fill_diagonal_(0)
        # [2N, 2N-1]
        logits = sim[mask].reshape(2*N, -1)
        labels = labels[mask].reshape(2*N, -1).nonzero()[:,-1]

        return logits, labels

    def calc_loss_2(self, z_i, z_j):
        N = z_i.shape[0] * self.world_size
        z = torch.cat((z_i, z_j), dim=0)  # [2N, D]
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.T

        sim_i_j = torch.diag(sim, N)
        sim_j_i = torch.diag(sim, -N)

        mask = torch.ones((2*N, 2*N), dtype=bool).fill_diagonal_(0)
        for i in range(N):
            mask[i, N + i] = 0
            mask[N + i, i] = 0

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(2*N, 1)
        negative_samples = sim[mask].reshape(2*N, -1)

        labels = torch.zeros(2*N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        return logits, labels


if __name__ == '__main__':

    import torchvision.models as models
    model = SimCLR(models.__dict__['resnet50'])

    z_i, z_j = torch.rand([4, 5]), torch.rand([4, 5])
    logitis, labels = model.calc_loss_1(z_i, z_j)
