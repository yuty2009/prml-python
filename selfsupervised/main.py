
import os
import sys; sys.path.append(os.path.dirname(__file__)+"/../")
import time
import math
import shutil
import random
import warnings
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.models as models
import torchvision.datasets as datasets

import moco
import simclr
import augment
import common.distributed as dist 


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='MoCo')
parser.add_argument('-d', '--dataset', default='ImageNet', metavar='DSET',
                    help='dataset used')
parser.add_argument('-r', '--dataset-dir', default='/home/public/datasets/ImageNet', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-o', '--output-dir', default='/home/yuty2009/tmp',
                    help='path where to save, empty for no saving')
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
    print("=> creating model '{}'".format(args.arch))

    # MoCo
    args.moco_dim = 128
    args.moco_k = 65536
    args.moco_m = 0.999
    args.moco_t = 0.07
    args.mlp = False
    args.aug_plus = False
    args.cos = False
    augtype = 'mocov1'
    model = moco.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    # SimCLR
    # augtype = 'simclr'
    # encoder = models.__dict__[args.arch]
    # n_features = encoder.fc.in_features
    # model = simclr.SimCLR(encoder=encoder, n_features=n_features)

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
    else:
        print("=> going to train from scratch")

    model = model.to(args.device)
    model = dist.convert_model(args, model)

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.dataset_dir))

    if args.dataset == "STL10":
        image_size = 224
        train_dataset = datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=augment.TransformContrast(
                augment.Augmentation.get(augtype, image_size)
                ),
        )
    elif args.dataset == "CIFAR10":
        image_size = 224
        train_dataset = datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=augment.TransformContrast(
                augment.Augmentation.get(augtype, image_size)
                ),
        )
    elif args.dataset == "ImageNet":
        image_size = 224
        train_dataset = datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=augment.TransformContrast(
                augment.Augmentation.get(augtype, image_size)
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

    # start training
    print("=> begin training")
    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_accu1, train_accu5, train_loss = train_epoch(
            train_loader, model, criterion, optimizer, epoch, args)

        if args.output_dir and epoch > 0 and epoch % args.save_freq == 0:
            if not args.distributed or args.rank == 0:
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, epoch, is_best=False, save_dir=args.output_dir)

        print(f"Epoch: {epoch}, "
              f"Train loss: {train_loss:.3f}, accu@1: {train_accu1:.3f}, accu@5 {train_accu5:.3f}, "
              f"Epoch time = {time.time() - start_time:.1f} s")


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    accus1, accus5, losses = [], [], []
    for i, (images, _) in enumerate(train_loader):

        images[0] = images[0].to(args.device)
        images[1] = images[1].to(args.device)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)
        accu1, accu5 = accuracy(output, target, topk=(1, 5))
        losses.append(loss.item())
        accus1.append(accu1.item())
        accus5.append(accu5.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f}"
                    .format(epoch, i, len(train_loader), loss.item(), accu1.item(), accu5.item()))

    return np.mean(accus1), np.mean(accus5), np.mean(losses)


def save_checkpoint(state, epoch, is_best, save_dir='./'):
    checkpoint_path = os.path.join(save_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch))
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
