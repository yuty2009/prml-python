# Refer to https://github.com/facebookresearch/swav/
import numpy as np
import torch
import torch.nn as nn


class SwAV(nn.Module):
    def __init__(
        self, encoder, encoder_dim=2048, feature_dim=512, dim=128, queue_size=0, num_prototypes=3000,
        temperature=0.1, num_crops=[2], crops_for_assign=[0, 1], sinkhorn_iters=3, epsilon=0.05):
        """
        encoder: encoder you want to use to get feature representations (eg. resnet50)
        encoder_dim: dimension of the encoder output (default: 2048 for resnets)
        feature_dim: intermediate dimension of the projector (default: 512)
        dim: projection dimension (default: 128)
        queue_size: queue size
        temperature: softmax temperature (default: 0.1)
        num_crops: list of number of crops (example: [2, 6])
        crops_for_assign: list of crops id used for computing assignments (default: [0, 1])
        num_prototypes: number of cluster centroids (default: 3000)
        sinkhorn_iters: number of iterations in Sinkhorn-Knopp algorithm
        epsilon: regularization parameter for Sinkhorn-Knopp algorithm
        """
        super(SwAV, self).__init__()

        self.queue_size = queue_size
        self.temperature = temperature
        self.num_crops = num_crops
        self.crops_for_assign = crops_for_assign
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon
        self.use_the_queue = False
        self.softmax = nn.Softmax(dim=1)

        self.encoder = encoder
        self.projector = nn.Sequential (
            nn.Linear(encoder_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, dim),
        )
        # prototype layer
        self.prototypes = None
        if isinstance(num_prototypes, list):
            self.prototypes = MultiPrototypes(dim, num_prototypes)
        elif num_prototypes > 0:
            self.prototypes = nn.Linear(dim, num_prototypes, bias=False)

        # create the queue
        if queue_size > 0:
            self.register_buffer("queue", torch.zeros(len(crops_for_assign), queue_size, dim))
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
        # compute Z^T C
        output = self.prototypes(z)
        return z, output

    def compute_loss(self, embedding, output, batch_size):
        loss = 0
        bs = batch_size
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()
                if self.queue_size > 0:
                    # time to use the queue
                    if self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.prototypes.weight.t()
                        ), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                # get assignments
                q = torch.exp(out / self.epsilon).t()
                q = distributed_sinkhorn(q, self.sinkhorn_iters)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.num_crops)), crop_id):
                p = self.softmax(output[bs * v: bs * (v + 1)] / self.temperature)
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.num_crops) - 1)
        loss /= len(self.crops_for_assign)

        return loss


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

