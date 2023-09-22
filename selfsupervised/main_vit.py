
import os
import json
import random
import datetime
import warnings
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory

import sys; sys.path.append(os.path.dirname(__file__)+"/../")
import common.distributed as dist
import common.torchutils as utils
from models.vit import ViT
from models.mlpmixer import MLPMixer
from models.convmixer import ConvMixer


parser = argparse.ArgumentParser(description='Vision Transformer')
parser.add_argument('-D', '--dataset', default='CIFAR10', metavar='PATH',
                    help='dataset used')
parser.add_argument('-d', '--dataset-dir', default='/home/yuty2009/data/prmldata/cifar10',
                    metavar='PATH', help='path to dataset')
parser.add_argument('-o', '--output-dir', default='/home/yuty2009/data/prmldata/cifar10/output/',
                    metavar='PATH', help='path where to save, empty for no saving')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit',
                    help='model architecture (default: vit)')
parser.add_argument('-p', '--patch-size', default=4, type=int, metavar='N',
                    help='patch size (default: 16) when dividing the long signal into windows')
parser.add_argument('--embed_dim', default=384, type=int, metavar='N',
                    help='embedded feature dimension (default: 192)')
parser.add_argument('--num_layers', default=6, type=int, metavar='N',
                    help='number of transformer layers (default: 6)')
parser.add_argument('--num_heads', default=8, type=int, metavar='N',
                    help='number of heads for multi-head attention (default: 6)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')
parser.add_argument('--global_pool', action='store_true', default=False)
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                    help='epochs to warmup LR')
parser.add_argument('--schedule', default='cos', type=str,
                    choices=['cos', 'step'],
                    help='learning rate schedule (how to change lr)')
parser.add_argument('--lr_drop', default=[0.6, 0.8], nargs='*', type=float,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('--topk', default=(1,), type=tuple,
                    help='top k accuracy')
parser.add_argument('-s', '--save-freq', default=50, type=int,
                    metavar='N', help='save frequency (default: 100)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate on the test dataset')
parser.add_argument('-r', '--resume',
                    default='',
                    type=str, metavar='PATH',
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
    print("=> loading dataset {} from '{}'".format(args.dataset, args.dataset_dir))

    def get_transforms(type='', size=224, mean_std=None):
        if mean_std is None:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            normalize = transforms.Normalize(
                mean=mean_std[0], std=mean_std[1])

        if type in ['', 'test', 'eval', 'val']:
            return transforms.Compose([
                transforms.Resize(size=size),
                transforms.ToTensor(),
                normalize
            ])
        elif type in ['train', 'training']:
            return transforms.Compose([
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])

    if str.lower(args.dataset) in ['cifar10', 'cifar-10']:
        args.num_classes = 10
        args.image_size = 32
        args.mean_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_dataset = datasets.CIFAR10(
            args.dataset_dir, train=True, download=True,
            transform=get_transforms('train', args.image_size, args.mean_std)
        )
        test_dataset = datasets.CIFAR10(
            args.dataset_dir, train=False, download=True,
            transform=get_transforms('test', args.image_size, args.mean_std)
        )
    elif str.lower(args.dataset) in ['stl10', 'stl-10']:
        args.num_classes = 10
        args.image_size = 96
        args.mean_std = ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237))
        train_dataset = datasets.STL10(
            args.dataset_dir, split="train", download=True,
            transform=get_transforms('train', args.image_size, args.mean_std)
        )
        test_dataset = datasets.STL10(
            args.dataset_dir, split="test", download=True,
            transform=get_transforms('test', args.image_size, args.mean_std)
        )
    elif str.lower(args.dataset) in ['imagenet', 'imagenet-1k', 'ilsvrc2012']:
        args.num_classes = 1000
        args.image_size = 224
        args.mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset = datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'train'),
            transform=get_transforms('train', args.image_size, args.mean_std)
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(args.dataset_dir, 'val'),
            transform=get_transforms('test', args.image_size, args.mean_std)
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
        test_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.arch in ['vit', 'ViT']:
        model = ViT(
            num_classes = args.num_classes,
            input_size = args.image_size,
            patch_size = args.patch_size,
            in_chans = 3,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            mlp_ratio = 4,
            droprate_embed = 0.1,
            pool='mean' if args.global_pool else 'cls',
        )
    elif args.arch in ['mlpmixer', 'MLPMixer']:
        model = MLPMixer(
            num_classes = args.num_classes,
            input_size = args.image_size,
            patch_size = args.patch_size,
            in_chans = 3,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            mlp_ratio_1 = 4.0,
            mlp_ratio_2 = 0.5,
            droprate_embed = 0.1,
        )
    elif args.arch in ['convmixer', 'ConvMixer']:
        model = ConvMixer(
            num_classes = args.num_classes,
            patch_size = args.patch_size,
            in_chans = 3,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            kernel_size = 9,
            droprate_embed = 0.1,
        )
    else:
        raise NotImplementedError

    criterion = nn.CrossEntropyLoss().to(args.device)
    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']
            state_dict = utils.convert_state_dict(checkpoint['state_dict'])
            if args.num_classes != state_dict['classifier.weight'].size(0):
                del state_dict['classifier.weight']
                del state_dict['classifier.bias']
                print("=> re-initialize fc layer")
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("=> going to train from scratch")

    model = dist.convert_model(args, model)

    if args.evaluate:
        test_loss, test_accu1 = utils.evaluate(
            test_loader, model, criterion, 0, args)
        print(f"Test loss: {test_loss:.4f} Acc@1: {test_accu1:.2f}")
        return

    args.writer = None
    if not args.distributed or args.rank == 0:
        with open(args.output_dir + "/args.json", 'w') as fid:
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            json.dump(args.__dict__, fid, indent=2, default=default)
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))

    print("=> begin training")
    args.best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        utils.adjust_learning_rate(optimizer, epoch, args)
        lr = optimizer.param_groups[0]["lr"]

        # train for one epoch
        train_loss, train_accu1 = utils.train_epoch(
            train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        test_loss, test_accu1 = utils.evaluate(
            test_loader, model, criterion, epoch, args)

        # remember best acc@1 and save checkpoint
        is_best = test_accu1 > args.best_acc
        args.best_acc = max(test_accu1, args.best_acc)

        if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
            if not args.distributed or args.rank == 0:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if args.ngpus > 1 else model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, epoch + 1,
                    is_best=is_best,
                    save_dir=os.path.join(args.output_dir, "checkpoint"))

        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar("Loss/train", train_loss, epoch)
            args.writer.add_scalar("Loss/test", test_loss, epoch)
            args.writer.add_scalar("Accu/train", train_accu1, epoch)
            args.writer.add_scalar("Accu/test", test_accu1, epoch)
            args.writer.add_scalar("Misc/learning_rate", lr, epoch)


if __name__ == '__main__':
    
    args = parser.parse_args()

    output_prefix = f"vit"
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
