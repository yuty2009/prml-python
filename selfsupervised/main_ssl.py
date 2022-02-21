
import os
import sys; sys.path.append(os.path.dirname(__file__)+"/../")
import time
import random
import warnings
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

import moco
import simclr
import augment
import common.distributed as dist 
from common.torchutils import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Self-supervised Learning Benchmarks')
parser.add_argument('--ssl', default='simclr', type=str,
                    help='self-supervised learning approach used')
parser.add_argument('-D', '--dataset', default='CIFAR10', metavar='PATH',
                    help='dataset used')
parser.add_argument('-d', '--dataset-dir', default='/home/yuty2009/data/cifar10', 
                    metavar='PATH', help='path to dataset')
parser.add_argument('-o', '--output-dir', default='/home/yuty2009/data/cifar10',
                    metavar='PATH', help='path where to save, empty for no saving')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
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
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default='cos', type=str,
                    choices=['cos', 'stepwise'],
                    help='learning rate schedule (how to change lr)')
parser.add_argument('--lr_drop', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--topk', default=(1, 5), nargs='*', type=int,
                    help='top k accuracy')
parser.add_argument('-v', '--verbose', default=True, type=bool,
                    help='whether print training information')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
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
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')


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
    print("=> loading dataset {} from '{}'".format(args.dataset, args.dataset_dir))

    if args.dataset in ['cifar10', 'cifar-10', 'CIFAR10', 'CIFAR-10']:
        args.image_size = 224
        train_dataset = datasets.CIFAR10(
            args.dataset_dir, download=True,
            transform=augment.TransformContrast(
                augment.get_transforms(args.ssl, args.image_size)
                ),
        )
    elif args.dataset in ['stl10', 'stl-10', 'STL10', 'STL-10']:
        args.image_size = 224
        train_dataset = datasets.STL10(
            args.dataset_dir, split="unlabeled", download=True,
            transform=augment.TransformContrast(
                augment.get_transforms(args.ssl, args.image_size)
                ),
        )
    elif args.dataset in ['imagenet', 'imagenet-1k', 'ImageNet', 'ImageNet-1k']:
        args.image_size = 224
        train_dataset = datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=augment.TransformContrast(
                augment.get_transforms(args.ssl, args.image_size)
                ),
        )
    else:
        raise NotImplementedError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    # create model
    print("=> creating model '{}'".format(args.arch))

    if args.ssl in ['mocov1', 'moco_v1', 'MoCov1', 'MoCo_v1']:
        args.moco_dim = 128
        args.moco_k = 65536
        args.moco_m = 0.999
        args.moco_t = 0.07
        args.mlp = False

        model = moco.MoCo(
            models.__dict__[args.arch],
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

    elif args.ssl in ['mocov2', 'moco_v2', 'MoCov2', 'MoCo_v2']:
        args.moco_dim = 128
        args.moco_k = 65536
        args.moco_m = 0.999
        args.moco_t = 0.07
        args.mlp = True

        model = moco.MoCo(
            models.__dict__[args.arch],
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

    elif args.ssl in ['simclr', 'simclr_v1', 'SimCLR', 'SimCLR_v1']:
        args.n_features = 2048
        model = simclr.SimCLR(
            models.__dict__[args.arch],
            n_features=args.n_features)
    
    else:
        raise NotImplementedError

    model = model.to(args.device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            state_dict = convert_state_dict(checkpoint['state_dict'])
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(args.device)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("=> going to train from scratch")

    model = dist.convert_model(args, model)

    args.writer = None
    if args.distributed and args.gpu != 0:
        args.writer = SummaryWriter(
            log_dir=os.path.join(args.output_dir, 'log')
            )

    # start training
    print("=> begin training")
    args.global_step = 0
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        lr = optimizer.param_groups[0]["lr"]

        # train for one epoch
        train_accu1, train_accu5, train_loss = train_epoch_ssl(
            train_loader, model, criterion, optimizer, epoch, args)

        if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
            if not args.distributed or args.rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if args.ngpus > 1 else model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, epoch + 1,
                    is_best=False, 
                    save_dir=os.path.join(args.output_dir, 'checkpoint'),
                    prefix=args.ssl)
        
        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar("Loss/train", train_loss, epoch)
            args.writer.add_scalar("Misc/learning_rate", lr, epoch)

        print(f"Epoch: {epoch} "
              f"Train loss: {train_loss:.4f} Acc@1: {train_accu1:.2f} Acc@5 {train_accu5:.2f} "
              f"Epoch time: {time.time() - start_time:.1f}s")


if __name__ == '__main__':

    args = parser.parse_args()

    if not hasattr(args, 'output_dir'):
        args.output_dir = args.dataset_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'log'))
        os.makedirs(os.path.join(args.output_dir, 'checkpoint'))

    args = dist.init_distributed_mode(args)
    if args.multiprocessing_distributed:
        if args.world_size > args.ngpus:
            print(f"Training with {args.world_size // args.ngpus} nodes, "
                  f"waiting until all nodes join before starting training")
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main, args=(args,), nprocs=args.ngpus, join=True)
    else:
        main(args.gpu, args)
