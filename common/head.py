import torch
import torch.nn as nn


class MLPHead(nn.Module):
    # Copy from DINOHead
    def __init__(
            self, in_dim, out_dim, n_layers=1, hidden_dims=[], 
            activation=nn.ReLU(inplace=True), use_bn=False, norm_last_layer=False
        ) -> None:
        super().__init__()
        n_layers = max(n_layers, 1)
        if n_layers == 1:
            self.projector = nn.Linear(in_dim, out_dim)
        else:
            assert len(hidden_dims) == n_layers - 1, 'hidden_dims must have (n_layers - 1) elements'
            if not use_bn:
                layers = [nn.Linear(in_dim, hidden_dims[0])]
            else:
                layers = [nn.Linear(in_dim, hidden_dims[0], bias=False)]
                layers.append(nn.BatchNorm1d(hidden_dims[0]))
            layers.append(activation)
            for i in range(n_layers - 2):
                if not use_bn:
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                else:
                    layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1], bias=False))
                    layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
                layers.append(activation)
            # output layer
            if norm_last_layer: # for dino
                last_layer = nn.utils.weight_norm(nn.Linear(hidden_dims[-1], out_dim, bias=False))
                last_layer.weight_g.data.fill_(1)
                last_layer.weight_g.requires_grad = False
                layers.append(last_layer)
            else:
                if not use_bn:
                    layers.append(nn.Linear(hidden_dims[-1], out_dim))
                else:
                    layers.append(nn.Linear(hidden_dims[-1], out_dim, bias=False))
                    layers.append(nn.BatchNorm1d(out_dim, affine=False))
            self.mlp = nn.Sequential(*layers)
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x
    