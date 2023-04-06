# Refer to https://github.com/facebookresearch/swav/
import torch
import torch.nn as nn


class DeepCluster(nn.Module):
    def __init__(
        self, encoder, encoder_dim=2048, feature_dim=512,
        n_mlplayers=2, hidden_dim=2048, use_bn=False,
        n_prototypes=3000):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 512)
        - n_mlplayers: number of MLP layers for the projector (default: 2)
        - hidden_dim: hidden dimension if a multi-layer projector was used (default: 2048)
        - use_bn: whether use batch normalization (default: False)
        - n_prototypes: number of cluster centroids (default: 3000)
        """
        super(DeepCluster, self).__init__()

        # create the online encoder
        self.encoder = encoder
        # create the online projector
        n_mlplayers = max(n_mlplayers, 1)
        activation = nn.ReLU(inplace=True)
        if n_mlplayers == 1:
            self.projector = nn.Linear(encoder_dim, feature_dim)
        else:
            if not use_bn:
                layers = [nn.Linear(encoder_dim, hidden_dim)]
            else:
                layers = [nn.Linear(encoder_dim, hidden_dim, bias=False)]
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            for _ in range(n_mlplayers - 2):
                if not use_bn:
                    layers.append(nn.Linear(hidden_dim, hidden_dim))
                else:
                    layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(activation)
            # output layer
            if not use_bn:
                layers.append(nn.Linear(hidden_dim, feature_dim))
            else:
                layers.append(nn.Linear(hidden_dim, feature_dim, bias=False))
                layers.append(nn.BatchNorm1d(feature_dim, affine=False))
            self.projector = nn.Sequential(*layers)
        # prototype layer
        self.prototypes = None
        if isinstance(n_prototypes, list):
            self.prototypes = MultiPrototypes(feature_dim, n_prototypes)
        elif n_prototypes > 0:
            self.prototypes = nn.Linear(feature_dim, n_prototypes, bias=False)

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
        # normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)
        # compute Z^T C
        output = self.prototypes(z)
        return output, z


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, n_prototypes):
        super(MultiPrototypes, self).__init__()
        self.n_heads = len(n_prototypes)
        for i, k in enumerate(n_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.n_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out