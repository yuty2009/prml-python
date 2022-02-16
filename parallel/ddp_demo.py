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
import shutil
import random
import warnings
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision

import common.distributed as dist


parser = argparse.ArgumentParser(description='PyTorch DistributedDataParallel Demo')
parser.add_argument('-d', '--data-dir', default='/home/yuty2009/data/cifar10', metavar='DIR',
                help='path to dataset')
parser.add_argument('-o', '--output_dir', default='/home/yuty2009/tmp',
                    help='path where to save, empty for no saving')
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
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-s', '--save-freq', default=100, type=int,
                    metavar='N', help='save frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
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

    # create model
    arch = 'resnet50'
    print("=> creating model '{}'".format(arch))
    model = torchvision.models.__dict__[arch](num_classes=10)
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

    # Data loading code
    print("=> loading data from '{}'".format(args.data_dir))
    traindir = os.path.join(args.data_dir, './')
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=traindir, train=True, download=True, transform=transform)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    valid_dataset = torchvision.datasets.CIFAR10(
        root=traindir, train=False, download=True, transform=transform)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("=> begin training")
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(train_loader, model, criterion, optimizer, epoch, args)
        
        accu1 = evaluate(valid_loader, model, criterion, args)

        if args.output_dir and epoch > 0 and epoch % args.save_freq == 0:
            if not args.distributed or args.rank == 0:
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, epoch, is_best=False, save_dir=args.output_dir)

        print("Train Epoch: {:03d} Accu: {:.4f}".format(epoch, accu1))


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    for i, (images, target) in enumerate(train_loader):

        images = images.to(args.device)
        target = target.to(args.device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f}"
                .format(epoch, i, len(train_loader), loss.item()))


def evaluate(eval_loader, model, criterion, args):
    model.eval()
    with torch.no_grad():
        correct = 0
        for i, (images, target) in enumerate(eval_loader):

            images = images.to(args.device)
            target = target.to(args.device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            preds = output.argmax(dim=1, keepdim=True)
            correct += target.eq(preds.view_as(target)).sum()

    return correct / len(eval_loader.dataset)


def save_checkpoint(state, epoch, is_best, save_dir='./'):
    checkpoint_path = os.path.join(save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)


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