
import os
import random
import datetime
import warnings
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchvision.models as models
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter

import sys; sys.path.append(os.path.dirname(__file__)+"/../")
import augment
import common.distributed as dist
import common.torchutils as utils
from engine_ssl import *
from models.vit import ViT


image_datasets = {
    'cifar10' : {
        'data_dir' : 'e:/prmldata/cifar10/',
        'output_dir' : 'e:/prmldata/cifar10/output/',
    },
    'stl10' : {
        'data_dir' : 'e:/prmldata/stl10/',
        'output_dir' : 'e:/prmldata/stl10/output/',
    },
    'imagenet' : {
        'data_dir' : '/home/yuty2009/data/prmldata/imagenet/',
        'output_dir' : '/home/yuty2009/data/prmldata/imagenet/output/',
    }
}

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Self-supervised Learning Benchmarks')
parser.add_argument('-D', '--dataset', default='cifar10', metavar='PATH',
                    help='dataset used')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit',
                    help='model architecture, default: resnet50')
parser.add_argument('--pretrained', 
                    default='e:/prmldata/cifar10/output/mae_vit/session_20230413145313/checkpoint/chkpt_0800.pth.tar',
                    metavar='PATH', help='path to pretrained model (default: none)')
parser.add_argument('--global_pool', action='store_true', default=False)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
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
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
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
parser.add_argument('--topk', default=(1, 5), type=tuple,
                    help='top k accuracy')
parser.add_argument('-s', '--save-freq', default=50, type=int,
                    metavar='N', help='save frequency (default: 100)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate on the test dataset')
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
parser.add_argument('--mp', '--mp-dist', action='store_true', default=False,
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
        args.num_classes = 10
        args.image_size = 32
        args.mean_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_dataset = datasets.CIFAR10(
            args.data_dir, train=True, download=True,
            transform=augment.get_transforms('train', args.image_size, args.mean_std)
        )
        test_dataset = datasets.CIFAR10(
            args.data_dir, train=False, download=True,
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )
    elif str.lower(args.dataset) in ['stl10', 'stl-10']:
        args.num_classes = 10
        args.image_size = 96
        args.mean_std = ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237))
        train_dataset = datasets.STL10(
            args.data_dir, split="train", download=True,
            transform=augment.get_transforms('train', args.image_size, args.mean_std)
        )
        test_dataset = datasets.STL10(
            args.data_dir, split="test", download=True,
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )
    elif str.lower(args.dataset) in ['imagenet', 'imagenet-1k', 'ilsvrc2012']:
        args.num_classes = 1000
        args.image_size = 224
        args.mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=augment.get_transforms('train', args.image_size, args.mean_std)
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'val'),
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
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

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch in models.__dict__.keys():
        base_encoder = models.__dict__[args.arch]()
        base_encoder = get_base_encoder(base_encoder, args)
    elif args.arch == 'vit':
        base_encoder = ViT(
            input_size = args.image_size,
            patch_size = 4,
            in_chans = 3,
            num_classes = 0, # make the classifier head to be nn.Identity()
            embed_dim = 384,
            num_layers = 6,
            num_heads = 8,
            mlp_ratio = 4,
            pool='mean' if args.global_pool else 'cls',
        )
    model = LinearClassifier(base_encoder, args.num_classes, freeze_encoder=True)

    model = model.to(args.device)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = get_optimizer(model, args)

    if args.pretrained:
        utils.load_checkpoint(args.pretrained, model, strict=False) # for contrastive pretraining
        # utils.load_checkpoint(args.pretrained, model.encoder, strict=False) # for masked pretraining

    model = dist.convert_model(args, model)

    if args.evaluate:
        test_loss, test_accu1, test_accu5 = utils.evaluate(
            test_loader, model, criterion, 0, args)
        print(f"Test loss: {test_loss:.4f} Acc@1: {test_accu1:.2f} Acc@5 {test_accu5:.2f}")
        return

    args.writer = None
    if not args.distributed or args.rank == 0:
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f"log"))

    # start training
    print("=> begin training")
    args.best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        utils.adjust_learning_rate(optimizer, epoch, args)
        lr = optimizer.param_groups[0]["lr"]

        # train for one epoch
        train_loss, train_accu1, train_accu5 = utils.train_epoch(
            train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        test_loss, test_accu1, test_accu5 = utils.evaluate(
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
                    save_dir=os.path.join(args.output_dir, f"checkpoint"))

        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar("Loss/train", train_loss, epoch)
            args.writer.add_scalar("Loss/test", test_loss, epoch)
            args.writer.add_scalar("Accu/train", train_accu1, epoch)
            args.writer.add_scalar("Accu/test", test_accu1, epoch)
            args.writer.add_scalar("Misc/learning_rate", lr, epoch)


if __name__ == '__main__':

    args = parser.parse_args()

    args.data_dir = image_datasets[args.dataset]['data_dir']
    args.output_dir = image_datasets[args.dataset]['output_dir']

    output_prefix = f"eval_{args.arch}"
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
