# Refer to https://github.com/facebookresearch/swav/
import torch
import torch.nn as nn
import os, sys; sys.path.append(os.getcwd())
from common.head import MLPHead


class SwAV(nn.Module):
    def __init__(self, encoder, encoder_dim=2048, feature_dim=512,
        n_mlplayers=2, hidden_dim=2048, use_bn=False,
        n_prototypes=3000, ncrops=2, queue_size=0):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 512)
        - n_mlplayers: number of MLP layers for the projector (default: 2)
        - hidden_dim: hidden dimension if a multi-layer projector was used (default: 2048)
        - use_bn: whether use batch normalization (default: False)
        - n_prototypes: number of cluster centroids (default: 3000)
        - ncrops: number of crops (example: 8)
        - queue_size: queue size (optional)
        """
        super(SwAV, self).__init__()

        self.ncrops = ncrops
        self.queue_size = queue_size

        # create the online encoder
        self.encoder = encoder
        # create the online projector
        self.projector = MLPHead(
            in_dim=encoder_dim, out_dim=feature_dim, n_layers=n_mlplayers,
            hidden_dims=[hidden_dim]*(n_mlplayers-1), use_bn=use_bn,
        )
        
        # prototype layer
        self.prototypes = None
        if isinstance(n_prototypes, list):
            self.prototypes = MultiPrototypes(feature_dim, n_prototypes)
        elif n_prototypes > 0:
            self.prototypes = nn.Linear(feature_dim, n_prototypes, bias=False)

        # create the queue
        if queue_size > 0:
            # we use the first 2 crops (global views) to get assignments
            self.register_buffer("queue", torch.zeros(2, queue_size, feature_dim))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # projection
        z = self.projector(output)
        # normalize feature embeddings
        z = nn.functional.normalize(z, dim=1)
        # normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)
        # compute Z^T C
        output = self.prototypes(z)
        # use queue if necessary
        if self.queue_size > 0:
            z = z.detach()
            output = self.update_queue(output, z)
        return output, z

    def update_queue(self, output, embedding):
        output = output.chunk(self.ncrops)
        embedding = embedding.chunk(self.ncrops)
        bs = output[0].size(0)
        output_new = []
        for i in range(2): # we use the first 2 crops to get assignments
            with torch.no_grad():
                out = output[i].detach()
                # time to use the queue
                if self.queue_size > 0 and not torch.all(self.queue[i, -1, :] == 0):
                    out = torch.cat((torch.mm(
                        self.queue[i],
                        self.prototypes.weight.t()
                    ), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[i]
                output_new.append(out)
        return torch.cat(output_new)


class SwAVLoss(nn.Module):
    def __init__(self, ncrops=2, temperature=0.1, sinkhorn_iters=3, epsilon=0.05):
        """
        temperature: softmax temperature (default: 0.1)
        sinkhorn_iters: number of iterations in Sinkhorn-Knopp algorithm
        epsilon: regularization parameter for Sinkhorn-Knopp algorithm
        """
        super(SwAVLoss, self).__init__()
        self.ncrops = ncrops
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon

    def forward(self, output, embedding):
        output = output.chunk(self.ncrops)
        embedding = embedding.chunk(self.ncrops)
        bs = embedding[0].size(0)
        total_loss = 0
        n_loss_terms = 0
        for i in range(2): # we use the first 2 crops to get assignments
            with torch.no_grad():
                out = output[i].detach()
                # get assignments
                q = torch.exp(out / self.epsilon).t()
                q = distributed_sinkhorn(q, self.sinkhorn_iters)[-bs:]
            # cluster assignment prediction
            for v in range(self.ncrops):
                if v == i:
                    # we skip cases where student and teacher operate on the same view
                    continue
                x = output[v] / self.temperature
                loss = torch.sum(-q * nn.functional.log_softmax(x, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss


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


@torch.no_grad()
def distributed_sinkhorn(Q, n_iters, world_size=1):
    B = Q.shape[1] * world_size # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    # dist.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(n_iters):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        # dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


@torch.no_grad()
def sinkhorn_knopp(Q, n_iters):
    """
    https://michielstock.github.io/posts/2017/2017-11-5-OptimalTransport/
    https://github.com/KeremTurgutlu/self_supervised/tree/main/self_supervised
    https://en.wikipedia.org/wiki/Sinkhorn%27s_theorem#Sinkhorn-Knopp_algorithm
    """
    Q /= Q.sum()
    r = (torch.ones(Q.shape[0]) / Q.shape[0]).to(Q.device)
    c = (torch.ones(Q.shape[1]) / Q.shape[1]).to(Q.device)

    for it in range(n_iters):
        u = Q.sum(1)
        Q *= (r / u).unsqueeze(1) # notice what * means in python
        Q *= (c / Q.sum(0)).unsqueeze(0)
    return (Q / Q.sum(0, keepdim=True)).t().float()

