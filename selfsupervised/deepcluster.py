# Refer to https://github.com/facebookresearch/swav/
import numpy as np
import torch
import torch.nn as nn


class DeepCluster(nn.Module):
    def __init__(
        self, encoder, encoder_dim=2048, feature_dim=512, dim=128, num_prototypes=3000):
        """
        encoder: encoder you want to use to get feature representations (eg. resnet50)
        encoder_dim: dimension of the encoder output (default: 2048 for resnets)
        feature_dim: intermediate dimension of the projector (default: 512)
        dim: projection dimension (default: 128)
        num_prototypes: number of cluster centroids (default: 3000)
        """
        super(DeepCluster, self).__init__()

        self.encoder = encoder
        self.projector = nn.Sequential(
                nn.Linear(encoder_dim, feature_dim, bias=False),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feature_dim, dim))
        self.model = nn.Sequential(self.encoder, self.projector)

        # prototype layer
        self.prototypes = None
        if isinstance(num_prototypes, list):
            self.prototypes = MultiPrototypes(dim, num_prototypes)
        elif num_prototypes > 0:
            self.prototypes = nn.Linear(dim, num_prototypes, bias=False)

    def forward(self, inputs):
        # encoding
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.encoder(torch.cat(inputs[start_idx: end_idx]))
            if start_idx == 0:
                out = _out
            else:
                out = torch.cat((out, _out))
            start_idx = end_idx
        # projection
        z = self.projector(out)
        # normalize feature embeddings
        z = nn.functional.normalize(z, dim=1)
        # compute Z^T C
        output = self.prototypes(z)
        return z, output


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, num_prototypes):
        super(MultiPrototypes, self).__init__()
        self.n_heads = len(num_prototypes)
        for i, k in enumerate(num_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.n_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out