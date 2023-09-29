# Copy from https://github.com/facebookresearch/moco/blob/main/moco/builder.py
# Refer to  https://github.com/leftthomas/SimCLR
#           https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
import copy
import torch
import torch.nn as nn
import os, sys; sys.path.append(os.getcwd())
from common import distributed as dist
from common.head import MLPHead


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, encoder, encoder_dim=2048, feature_dim=512,
        n_mlplayers=2, hidden_dim=2048, use_bn=False,
        queue_size=65536, momentum=0.999, symmetric=False):
        """
        - encoder: encoder you want to use to get feature representations (eg. resnet50)
        - encoder_dim: dimension of the encoder output (default: 2048 for resnets)
        - feature_dim: dimension of the projector output (default: 512)
        - n_mlplayers: number of MLP layers for the projector (default: 2)
        - hidden_dim: hidden dimension if a multi-layer projector was used (default: 2048)
        - use_bn: whether use batch normalization (default: False)
        - queue_size: queue size; number of negative keys (default: 65536)
        - momentum: moco momentum of updating key encoder (default: 0.999)
        - symmetric: use symmetric loss or not (default: False)
        """
        super(MoCo, self).__init__()

        self.momentum = momentum
        self.queue_size = queue_size
        self.symmetric = symmetric

        # create the online encoder
        self.encoder = encoder
        # create the online projector
        self.projector = MLPHead(
            in_dim=encoder_dim, out_dim=feature_dim, n_layers=n_mlplayers,
            hidden_dims=[hidden_dim]*(n_mlplayers-1), use_bn=use_bn,
        )

        self.model = nn.Sequential(self.encoder, self.projector)
        self.model_momentum = copy.deepcopy(self.model)
        for p in self.model_momentum.parameters():
            p.requires_grad = False

        # create the queue
        self.register_buffer("queue", torch.randn(feature_dim, queue_size))
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
        keys = dist.all_gather(keys)

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
        x_gather = dist.all_gather(x)
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
        x_gather = dist.all_gather(x)
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
