# Copy from https://github.com/facebookresearch/moco/blob/main/moco/builder.py
# Refer to  https://github.com/leftthomas/SimCLR
#           https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
import copy
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(
        self, encoder, encoder_dim=2048, feature_dim=512, dim=128, num_mlplayers=1,
        queue_size=65536, momentum=0.999, symmetric=False):
        """
        encoder: encoder you want to use to get feature representations (eg. resnet50)
        encoder_dim: dimension of the encoder output (default: 2048 for resnets)
        feature_dim: intermediate dimension of the projector (default: 512)
        dim: projection dimension (default: 128)
        queue_size: queue size; number of negative keys (default: 65536)
        momentum: moco momentum of updating key encoder (default: 0.999)
        temperature: softmax temperature (default: 0.07)
        mlp: with a multi-layer projector (default: False)
        symmetric: use symmetric loss or not (default: False)
        """
        super(MoCo, self).__init__()

        self.momentum = momentum
        self.queue_size = queue_size
        self.symmetric = symmetric

        self.encoder = encoder
        if num_mlplayers == 1:
            self.projector = nn.Linear(encoder_dim, dim) # output layer
        elif num_mlplayers == 2:
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

        self.model = nn.Sequential(self.encoder, self.projector)
        self.model_momentum = copy.deepcopy(self.model)
        for p in self.model_momentum.parameters():
            p.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.model_momentum.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        if self.symmetric:
            q1, k1 = self._forward_1(im_q, im_k)
            q2, k2 = self._forward_1(im_k, im_q)
            k = torch.cat([k1, k2], dim=0)
            # dequeue and enqueue
            self._dequeue_and_enqueue(k)
            return q1, q2, k1, k2, self.queue
        else:
            q, k = self._forward_1(im_q, im_k)
            # dequeue and enqueue
            self._dequeue_and_enqueue(k)
            return q, k, self.queue

    def _forward_1(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q = self.model(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            k = self.model_momentum(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
        return q, k


class NTXentLossWithQueue(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLossWithQueue, self).__init__()
        self.temperature = temperature

    def forward(self, q, k, queue):
        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.temperature
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
