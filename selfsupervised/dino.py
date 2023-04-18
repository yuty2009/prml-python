# Refer to: https://github.com/facebookresearch/dino/
#           https://github.com/KeremTurgutlu/self_supervised/
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import common.distributed as dist


class DINO(nn.Module):
    def __init__(
        self, encoder, encoder_dim=2048, feature_dim=256,
        n_mlplayers=3, hidden_dim=2048, bottleneck_dim=256,
        use_bn=False, norm_last_layer=True, momentum=0.999):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output, your feature dimension (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 256)
        - n_mlplayers: number of MLP layers for the projector (default: 3)
        - hidden_dim: hidden dimension if a multi-layer projector was used
        - bottleneck_dim: bottleneck dimension if a multi-layer projector was used
        - use_bn: whether use batch normalization (default: False)
        - norm_last_layer: whether or not to weight normalize the last layer of the DINO head.
        - momentum: momentum of updating key encoder (default: 0.999)
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.
        (default: True)
        """
        super(DINO, self).__init__()

        self.momentum = momentum

        # create the online encoder
        self.encoder = encoder
        self.head_s = DINOHead(
            in_dim=encoder_dim, out_dim=feature_dim, use_bn=use_bn, norm_last_layer=norm_last_layer, 
            nlayers=n_mlplayers, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim,
        )
        self.head_t = DINOHead(
            in_dim=encoder_dim, out_dim=feature_dim, use_bn=use_bn, norm_last_layer=True, 
            nlayers=n_mlplayers, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim,
        )
        self.student = nn.Sequential(encoder, self.head_s)
        self.teacher = nn.Sequential(copy.deepcopy(encoder), self.head_t)
        # disable backpropagation through the teacher
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _momentum_update_teacher(self):
        for param_s, param_t in zip(self.student.parameters(),
                                    self.teacher.parameters()):
            param_t.data = param_t.data * self.momentum + param_s.data * (1. - self.momentum)

    def forward(self, inputs):
        # EMA update for the teacher
        with torch.no_grad():
            self._momentum_update_teacher()
        # multi-crop forward
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out_s = self.student(torch.cat(inputs[start_idx: end_idx]))
            if start_idx == 0:
                output_s = _out_s
            else:
                output_s = torch.cat((output_s, _out_s))
            start_idx = end_idx
        # only the 2 global views pass through the teacher
        output_t = self.teacher(torch.cat(inputs[:2]))
        # normalize the outputs
        output_s = nn.functional.normalize(output_s, dim=1)
        output_t = nn.functional.normalize(output_t, dim=1)
        return output_s, output_t


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops=2, temperature_s=0.1, temperature_t=0.07, center_momentum=0.9):
        super().__init__()
        self.ncrops = ncrops
        self.temperature_s = temperature_s
        self.temperature_t = temperature_t
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, output_s, output_t):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        out_s = output_s / self.temperature_s
        out_s = out_s.chunk(self.ncrops)

        # teacher centering and sharpening
        out_t = F.softmax((output_t - self.center) / self.temperature_t, dim=-1)
        out_t = out_t.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(out_t):
            for v in range(len(out_s)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(out_s[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(output_t)
        return total_loss

    @torch.no_grad()
    def update_center(self, output_t):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(output_t, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        # batch_center = batch_center / (len(output_t) * dist.get_world_size())
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOHead(nn.Module):
    '''
    copy.deepcopy:
    RuntimeError: Only Tensors created explicitly by the user (graph leaves)
    support the deepcopy protocol at the moment
    https://pytorch.org/docs/stable/generated/torch.nn.utils.weight_norm.html
    https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    '''
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
