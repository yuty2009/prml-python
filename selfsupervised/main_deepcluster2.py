
import os
import json
import random
import datetime
import warnings
import argparse
import numpy as np
from scipy.sparse import csr_matrix

import torch
import torch.multiprocessing as mp
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import sys; sys.path.append(os.path.dirname(__file__)+"/../")
import common.distributed as dist
import common.torchutils as utils
from augment import *
from sslutils import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Self-Supervised Learning Benchmarks')
parser.add_argument('--ssl', default='deepcluster2', type=str,
                    help='self-supervised learning approach used')
parser.add_argument('-D', '--dataset', default='CIFAR10', metavar='PATH',
                    help='dataset used')
parser.add_argument('-d', '--data-dir', default='/home/yuty2009/data/cifar10', 
                    metavar='PATH', help='path to dataset')
parser.add_argument('-o', '--output-dir', default='/home/yuty2009/data/cifar10',
                    metavar='PATH', help='path where to save, empty for no saving')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default='sgd', type=str,
                    choices=['adam', 'adamw', 'sgd', 'lars'],
                    help='optimizer used to learn the model')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default='step', type=str,
                    choices=['cos', 'step'],
                    help='learning rate schedule (how to change lr)')
parser.add_argument('--lr_drop', default=[0.6, 0.8], nargs='*', type=float,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--topk', default=(1, 5), nargs='*', type=int,
                    help='top k accuracy')
parser.add_argument('-s', '--save-freq', default=50, type=int,
                    metavar='N', help='save frequency (default: 100)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--mp', '--mp-dist', action='store_true',
                    help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training',
                    dest='mp_dist')


def main(gpu, args):
    args.gpu = gpu
    args = dist.init_distributed_process(args)

    if args.seed is not None:
        if args.gpu is not None:
            args.seed += args.gpu
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
    if str.lower(args.dataset) in ['cifar10', 'cifar-10']:
        args.size_crops = [32]
        args.num_crops = [2]
        args.crops_for_assign = [0, 1]
        args.image_size = 32
        args.mean_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_dataset = datasets.CIFAR10(
            args.data_dir, train=True, download=True,
        )
        memory_dataset = datasets.CIFAR10(
            args.data_dir, train=True, download=True,
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )
        test_dataset = datasets.CIFAR10(
            args.data_dir, train=False, download=True,
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )

    elif str.lower(args.dataset) in ['stl10', 'stl-10']:
        args.size_crops = [96]
        args.num_crops = [2]
        args.crops_for_assign = [0, 1]
        args.image_size = 96
        args.mean_std = ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237))
        train_dataset = datasets.STL10(
            args.data_dir, split="train", download=True,
        )
        memory_dataset = datasets.STL10(
            args.data_dir, split="train", download=True,
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )
        test_dataset = datasets.STL10(
            args.data_dir, split="test", download=True,
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )

    elif str.lower(args.dataset) in ['imagenet', 'imagenet-1k', 'ilsvrc2012']:
        args.size_crops = [224]
        args.num_crops = [2]
        args.crops_for_assign = [0, 1]
        args.image_size = 224
        args.mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
        )
        memory_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'val'),
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )

    multicrop_trans = get_multicrop_transforms(args.size_crops, args.num_crops, mean_std=args.mean_std)
    train_dataset = MultiCropDataset(train_dataset, multicrop_trans, return_index=True)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=500, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=500, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    base_encoder = models.__dict__[args.arch]()
    base_encoder = get_base_encoder(base_encoder, args)
    model, criterion = get_ssl_model_and_criterion(base_encoder, args)
    # print(model)
    optimizer = get_optimizer(model, args)

    # optionally resume from a checkpoint
    if hasattr(args, 'resume') and args.resume:
        utils.load_checkpoint(args.resume, model, optimizer, args)
    else:
        print("=> going to train from scratch")

    model = dist.convert_model(args, model)
    torch.backends.cudnn.benchmark = True

    args.knn_k = 200
    args.knn_t = 0.1

    args.writer = None
    if not args.distributed or args.rank == 0:
        with open(args.output_dir + "/args.json", 'w') as fid:
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            json.dump(args.__dict__, fid, indent=2, default=default)
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))

    mb_index, mb_embeddings = init_memory(train_loader, model, args)

    # start training
    print("=> begin training")
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        utils.adjust_learning_rate(optimizer, epoch, args)
        lr = optimizer.param_groups[0]["lr"]

        assignments = cluster_memory(
            model, mb_index, mb_embeddings, 
            len(train_loader.dataset), args.dim, args.num_prototypes, args=args)
        print("Clustering for epoch {} done.".format(epoch))

        # train for one epoch
        train_loss, mb_index, mb_embeddings = train_epoch(
            train_loader, model, criterion, optimizer, epoch, args, 
            mb_index, mb_embeddings, assignments)
        # evaluate for one epoch
        real_model = model.module if args.ngpus > 1 else model
        test_accu1, test_accu5 = evaluate_ssl(memory_loader, test_loader, real_model.encoder, epoch, args)

        if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
            if not args.distributed or args.rank == 0:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': real_model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, epoch + 1,
                    is_best=False,
                    save_dir=os.path.join(args.output_dir, f"checkpoint"))
        
        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar("Loss/train", train_loss, epoch)
            args.writer.add_scalar("Accu/test", test_accu1, epoch)
            args.writer.add_scalar("Misc/learning_rate", lr, epoch)


def train_epoch(data_loader, model, criterion, optimizer, epoch, args,
                mb_index, mb_embeddings, assignments):
    model.train()

    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True
    data_bar = tqdm.tqdm(data_loader) if show_bar else data_loader
    
    start_idx = 0
    total_loss = 0.0
    for it, (index, inputs) in enumerate(data_bar):
        iteration = epoch * len(data_bar) + it

        bs = inputs[0].size(0)
        index = index.to(args.device)
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(args.device)

        outputs, embeddings = model(inputs)

        outputs /= args.temperature
        targets = assignments[index].repeat(len(inputs)).to(args.device)
        loss = criterion(outputs, targets)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # cancel gradients for the prototypes
        if iteration < args.freeze_prototypes_niters:
            for name, p in model.named_parameters():
                if "prototypes" in name:
                    p.grad = None

        optimizer.step()

        # ============ update memory banks ... ============
        embeddings.detach()
        mb_index[start_idx : start_idx + bs] = index
        for i, crop_idx in enumerate(args.crops_for_assign):
            mb_embeddings[i][start_idx : start_idx + bs] = \
                embeddings[crop_idx * bs : (crop_idx + 1) * bs]
        start_idx += bs

        loss = dist.all_reduce(loss)
        total_loss += loss.item()

        if show_bar:
            data_bar.set_description(
                "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f}".format(
                    epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / len(data_loader)))

    return total_loss / len(data_loader), mb_index, mb_embeddings


def init_memory(data_loader, model, args):
    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True

    mb_size = len(data_loader) * args.batch_size
    mb_index = torch.zeros(mb_size).long().to(args.device)
    mb_embeddings = torch.zeros(len(args.crops_for_assign), mb_size, args.dim).to(args.device)

    start_idx = 0
    with torch.no_grad():
        data_bar = tqdm.tqdm(data_loader, desc='Init memory bank') if show_bar else data_loader
        for index, inputs in data_bar:
            bs = inputs[0].size(0)
            index = index.to(args.device)
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(args.device)
            outputs, embeddings = model(inputs)
            # fill the memory bank
            mb_index[start_idx : start_idx + bs] = index
            for i, crop_idx in enumerate(args.crops_for_assign):
                mb_embeddings[i][start_idx : start_idx + bs] = \
                    embeddings[crop_idx * bs : (crop_idx + 1) * bs]
            start_idx += bs
    
    return mb_index, mb_embeddings


def cluster_memory(
    model, mb_index, mb_embeddings,
    n_data, feature_dim, num_prototypes, n_iters=10, args=None):
    real_model = model.module if args.ngpus > 1 else model
    j = 0
    assignments = -100 * torch.ones(n_data).long()
    with torch.no_grad():
        K = num_prototypes
        # run distributed k-means

        # init centroids with elements from memory bank of rank 0
        centroids = torch.empty(K, feature_dim).to(args.device)
        if args.rank == 0:
            random_idx = torch.randperm(len(mb_embeddings[j]))[:K]
            assert len(random_idx) >= K, "please reduce the number of centroids"
            centroids = mb_embeddings[j][random_idx]
        dist.broadcast(centroids, 0)

        for it in range(n_iters + 1):

            # E step
            dot_products = torch.mm(mb_embeddings[j], centroids.t())
            _, local_assignments = dot_products.max(dim=1)

            # finish
            if it == n_iters: break

            # M step
            where_helper = get_indices_sparse(local_assignments.cpu().numpy())
            counts = torch.zeros(K).to(args.device).int()
            emb_sums = torch.zeros(K, feature_dim).to(args.device)
            for k in range(len(where_helper)):
                if len(where_helper[k][0]) > 0:
                    emb_sums[k] = torch.sum(
                        mb_embeddings[j][where_helper[k][0]],
                        dim=0,
                    )
                    counts[k] = len(where_helper[k][0])
            dist.all_reduce(counts)
            mask = counts > 0
            dist.all_reduce(emb_sums)
            centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

            # normalize centroids
            centroids = nn.functional.normalize(centroids, dim=1, p=2)

        if isinstance(num_prototypes, list):
            getattr(real_model.prototypes, "prototypes"+str(j)).weight.copy_(centroids)
        else:
            real_model.prototypes.weight.copy_(centroids)

        # gather the assignments
        assignments_all = dist.all_gather(local_assignments)
        assignments_all = assignments_all.cpu()

        # gather the indexes
        indexes_all = dist.all_gather(mb_index)
        indexes_all = indexes_all.cpu()

        # log assignments
        assignments[indexes_all] = assignments_all

        # next memory bank to use
        j = (j + 1) % len(args.crops_for_assign)

    return assignments


def get_indices_sparse(data):
    cols = np.arange(data.size)
    M = csr_matrix((cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size))
    return [np.unravel_index(row.data, data.shape) for row in M]


if __name__ == '__main__':

    args = parser.parse_args()

    output_prefix = f"ssl_{args.ssl}_{args.arch}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    args = dist.init_distributed_mode(args)
    if args.mp_dist:
        if args.world_size > args.ngpus:
            print(f"Training with {args.world_size // args.ngpus} nodes, "
                  f"waiting until all nodes join before starting training")
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main, args=(args,), nprocs=args.ngpus, join=True)
    else:
        main(args.gpu, args)
