"""
Run this demo by one of the following commands
1. Using a single GPU
    $ python ddp_demo.py --gpu 0
2. Using all available GPUs with DataParallel 
    $ python ddp_demo.py
3. Using all available GPUs on one node with multi-process DistributedDataParallel
    $ python ddp_demo.py --multiprocessing-distributed
4. Using all available GPUs on two nodes with multi-process DistributedDataParallel
    $ python ddp_demo.py --multiprocessing-distributed \
             --dist-url 'tcp://gpu01:23456' --world-size 2 --rank 0
    $ python ddp_demo.py --multiprocessing-distributed \
             --dist-url 'tcp://gpu01:23456' --world-size 2 --rank 1
5. Using all available GPUs on one node with DistributedDataParallel
    $ python -m torch.distributed.launch --nproc_per_node 8 ddp_demo.py
6. Using all available GPUs on two nodes with DistributedDataParallel
    $ python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node 8 \
             --master_addr gpu01 --master_port 23456 ddp_demo.py
    $ python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node 8 \
             --master_addr gpu01 --master_port 23456 ddp_demo.py
7. Using part of the GPUs with DistributedDataParallel
    $ CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 main.py
    $ CUDA_VISIBLE_DEVICES="4,5,6,7" python -m torch.distributed.launch --nproc_per_node 4 main.py
8. Using SLURM on a single node
    $ srun --partition=T4 --nodelist=gpu02 -n1 --gres=gpu:8 --ntasks-per-node=8 \
           python parallel/ddp_demo.py --multiprocessing-distributed
9. Using SLURM on multiple nodes
    $ srun --partition=T4 --nodelist=gpu[02-03] -n2 --gres=gpu:8 --ntasks-per-node=8 \
           python parallel/ddp_demo.py --multiprocessing-distributed --dist-url 'tcp://gpu02:23456'
"""
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
import torchvision.transforms as transforms

import common.distributed as dist
from common.torchutils import *


parser = argparse.ArgumentParser(description='PyTorch DistributedDataParallel Demo')
parser.add_argument('-d', '--dataset', default='CIFAR10', metavar='PATH',
                    help='dataset used')
parser.add_argument('-r', '--dataset-dir', default='/home/yuty2009/data/cifar10',
                    metavar='PATH', help='path to dataset')
parser.add_argument('-o', '--output-dir', default='/home/yuty2009/data/cifar10/checkpoint',
                    metavar='PATH', help='path where to save, empty for no saving')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default='', type=str,
                    choices=['cos', 'stepwise'],
                    help='learning rate schedule (how to change lr)')
parser.add_argument('--lr_drop', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--topk', default=(1, 5), type=tuple,
                    help='top k accuracy')
parser.add_argument('-v', '--verbose', default=True, type=bool,
                    help='whether print training information')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-s', '--save-freq', default=50, type=int,
                    metavar='N', help='save frequency (default: 100)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate on the test dataset')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
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

    def get_transforms(type='', size=224):
        if type in ['', 'test', 'eval', 'val']:
            return transforms.Compose([
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif type in ['train', 'training']:
            return transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    if args.dataset in ['cifar10', 'cifar-10', 'CIFAR10', 'CIFAR-10']:
        args.num_classes = 10
        args.image_size = 32
        train_dataset = datasets.CIFAR10(
            args.dataset_dir, train=True, download=True,
            transform=get_transforms('train', args.image_size)
        )
        test_dataset = datasets.CIFAR10(
            args.dataset_dir, train=False, download=True,
            transform=get_transforms('test', args.image_size)
        )
    elif args.dataset in ['stl10', 'stl-10', 'STL10', 'STL-10']:
        args.num_classes = 10
        args.image_size = 96
        train_dataset = datasets.STL10(
            args.dataset_dir, split="train", download=True,
            transform=get_transforms('train', args.image_size)
        )
        test_dataset = datasets.STL10(
            args.dataset_dir, split="test", download=True,
            transform=get_transforms('test', args.image_size)
        )
    elif args.dataset in ['imagenet', 'imagenet-1k', 'ImageNet', 'ImageNet-1k']:
        args.num_classes = 1000
        args.image_size = 224
        train_dataset = datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=get_transforms('train', args.image_size)
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'val'),
            transform=get_transforms('test', args.image_size)
        )
    else:
        raise NotImplementedError

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=10)
    # print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model = model.to(args.device)
    model = dist.convert_model(args, model)

    print("=> begin training")
    best_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_accu1, train_accu5, train_loss = train_epoch(
            train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        test_accu1, test_accu5, test_loss = evaluate(test_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = test_accu1 > best_acc1
        best_acc1 = max(test_accu1, best_acc1)

        if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
            if not args.distributed or args.rank == 0:
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, epoch, is_best=is_best, save_dir=args.output_dir, prefix=args.arch)

        print(f"Epoch: {epoch} "
              f"Train loss: {train_loss:.4f} Acc@1: {train_accu1:.2f} Acc@5 {train_accu5:.2f} "
              f"Test loss: {test_loss:.4f} Acc@1: {test_accu1:.2f} Acc@5 {test_accu5:.2f} "
              f"Epoch time: {time.time() - start_time:.1f}s")


if __name__ == '__main__':
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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