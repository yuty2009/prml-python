# Refer to: https://github.com/facebookresearch/dino/
#           https://github.com/KeremTurgutlu/self_supervised/
import math
import copy
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class DINO(nn.Module):
    def __init__(
        self, encoder, encoder_dim=2048, feature_dim=2048, dim=512, num_mlplayers=2,
        momentum=0.999, norm_last_layer=True):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output, your feature dimension (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 2048)
        - dim: hidden dimension of the predictor (default: 512)
        - num_mlplayers: number of MLP layers for the projector (default: 2)
        - momentum: momentum of updating key encoder (default: 0.999)
        - norm_last_layer: whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.
        (default: True)
        """
        super(DINO, self).__init__()

        self.momentum = momentum

        # create the online encoder
        self.encoder = encoder
        # create the online projector
        if num_mlplayers == 2:
            self.projector = nn.Sequential(nn.Linear(encoder_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim, affine=False)) # output layer
        elif num_mlplayers == 3:
            self.projector = nn.Sequential(nn.Linear(encoder_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(feature_dim, feature_dim, bias=False),
                                        nn.BatchNorm1d(feature_dim, affine=False)) # output layer
        # create student and teacher
        self.student = nn.Sequential(self.encoder, self.projector)
        self.teacher = copy.deepcopy(self.student)
        # disable backpropagation through the teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        # create the student predictor
        self.predictor_s = nn.utils.weight_norm(nn.Linear(feature_dim, dim, bias=False))
        self.predictor_s.weight_g.data.fill_(1)
        if norm_last_layer:
            self.predictor_s.weight_g.requires_grad = False
        # create the teacher predictor
        self.predictor_t = nn.utils.weight_norm(nn.Linear(feature_dim, dim, bias=False))
        self.predictor_t.weight_g.data.fill_(1)
        # self.predictor_t.weight_g.requires_grad = False
        for p in self.predictor_t.parameters():
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
        # last_layer (predictor)
        output_s = self.predictor_s(output_s)
        output_t = self.predictor_s(output_t)
        return output_s, output_t


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, temperature_s, temperature_t, center_momentum=0.9):
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
        batch_center = batch_center / (len(output_t) * dist.get_world_size())

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
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)