
import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, symmetric=False):
        super(NTXentLoss, self).__init__()
        self.symmetric = symmetric
        self.temperature = temperature
        self.world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()

    def forward(self, outputs):
        z1, z2 = outputs
        if not self.symmetric:
            return self.forward_1(z1, z2)
        else:
            loss_1 = self.forward_1(z1, z2)
            loss_2 = self.forward_1(z2, z1)
            loss = 0.5 * (loss_1 + loss_2)
            return loss
        
    def forward_1(self, z1, z2):
        N = z1.shape[0] # * self.world_size
        z = torch.cat((z1, z2), dim=0)  # [2N, D]
        # z = torch.cat(GatherLayer.apply(z), dim=0)
        # [2N, 2N]
        sim = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)
        idx = torch.arange(2*N).roll(N)
        labels = torch.eye(2*N, device=sim.device)[idx]

        mask = torch.ones((2*N, 2*N), dtype=bool).fill_diagonal_(0)
        # [2N, 2N-1]
        logits = sim[mask].reshape(2*N, -1)
        labels = labels[mask].reshape(2*N, -1).nonzero(as_tuple=False)[:,-1]
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


class NTXentLoss1(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size() 

    def forward(self, z1, z2):
        N = z1.shape[0] # * self.world_size
        z = torch.cat((z1, z2), dim=0)  # [2N, D]
        # z = torch.cat(GatherLayer.apply(z), dim=0)

        # sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim = torch.exp(torch.mm(z, z.t().contiguous()) / self.temperature)

        sim_12 = torch.diag(sim, N)
        sim_21 = torch.diag(sim, -N)

        mask = torch.ones((2*N, 2*N), dtype=bool).fill_diagonal_(0)
        for i in range(N):
            mask[i, N + i] = 0
            mask[N + i, i] = 0

        positive_samples = torch.cat((sim_12, sim_21), dim=0).reshape(2*N, 1)
        negative_samples = sim[mask].reshape(2*N, -1)

        labels = torch.zeros(2*N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = nn.functional.cross_entropy(logits, labels)
        return loss


class NTXentLoss2(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        N = z1.shape[0]
        # [2*B, D]
        out = torch.cat([z1, z2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * N, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * N, -1)
        # compute loss
        pos_sim = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss
    

class NTXentLossWithQueue(nn.Module):
    def __init__(self, temperature=0.5, symmetric=False):
        super(NTXentLossWithQueue, self).__init__()
        self.symmetric = symmetric
        self.temperature = temperature

    def forward(self, outputs):
        if not self.symmetric:
            q, k, queue = outputs
            return self.forward_1(q, k, queue)
        else:
            p1, p2, t1, t2, queue = outputs
            loss_1 = self.forward_1(p1, t1, queue)
            loss_2 = self.forward_1(p2, t2, queue)
            loss = 0.5 * (loss_1 + loss_2)
            return loss

    def forward_1(self, q, k, queue):
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
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        loss = nn.functional.cross_entropy(logits, labels)
        return loss
    

class MultiViewNTXentLossWithQueue(nn.Module):
    def __init__(self, loss_self, loss_cross, alpha=0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.loss_self = loss_self
        self.loss_cross = loss_cross
    def forward(self, p1, p2, t1, t2, queue_1, queue_2):
        loss_1 = self.loss_self(p1, t1, queue_1)
        loss_2 = self.loss_self(p2, t2, queue_2)
        loss_x = self.loss_cross(p1, p2)
        loss = self.alpha * 0.5 * (loss_1 + loss_2) + (1-self.alpha) * loss_x
        return loss
    

class CosSimLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CosineSimilarity(dim=1)

    def forward(self, outputs):
        p1, p2, t1, t2 = outputs
        loss = -0.5 * (self.loss_fn(p1, t2).mean() + self.loss_fn(p2, t1).mean())
        return loss
    

class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    https://github.com/open-mmlab/OpenSelfSup/blob/696d04950e55d504cf33bc83cfadbb4ece10fbae/openselfsup/models/utils/gather_layer.py
    '''
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[torch.distributed.get_rank()]
        return grad_out
    