
import os
import json
import random
import datetime
import warnings
import argparse
import numpy as np

import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory

import sys; sys.path.append(os.path.dirname(__file__)+"/../")
import common.distributed as dist
import common.torchutils as utils
from engine_ssl import *
from models.vit import ViT
from models.mae import MAE
from models.simmim import SimMIM
from models.mbyol import MBYOL
from models.mdino import MDINO
from models.cgpt import CGPTARModel


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

parser = argparse.ArgumentParser(description='Self-supervised Pretraining')
parser.add_argument('-D', '--dataset', default='cifar10', metavar='PATH',
                    help='dataset used')
parser.add_argument('--ssl', default='cgpt', type=str,
                    help='self-supervised learning approach used')
parser.add_argument('-p', '--patch-size', default=4, type=int, metavar='N',
                    help='patch size (default: 16) when dividing the long signal into windows')
parser.add_argument('--embed_dim', default=64, type=int, metavar='N',
                    help='embedded feature dimension (default: 192)')
parser.add_argument('--num_layers', default=6, type=int, metavar='N',
                    help='number of transformer layers (default: 6)')
parser.add_argument('--num_heads', default=8, type=int, metavar='N',
                    help='number of heads for multi-head attention (default: 6)')
parser.add_argument('--mask_prob', default=0.75, type=float,
                    help='Masking ratio (percentage of removed patches).')
parser.add_argument('--norm_pix_loss', default=False, action='store_true',
                    help='Use (per-patch) normalized pixels as targets for computing loss')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
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
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
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
parser.add_argument('--mp', '--mp-dist', action='store_true', default=False,
                    help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training',
                    dest='mp_dist')
parser.add_argument('--use_amp', action='store_true', default=True,
                    help='Use mixed precision training')


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
    train_dataset, memory_dataset, test_dataset = get_train_dataset(args, constrastive=False, evaluate=True)

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
    print(f"=> creating {args.ssl} model")
    encoder = ViT(
        input_size = args.image_size,
        patch_size = args.patch_size,
        embed_dim = args.embed_dim,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        mlp_ratio = 4.0,
        pool = 'none',
    )

    if args.ssl in ['mae', 'MAE']:
        model = MAE(
            input_size = args.image_size,
            patch_size = args.patch_size,
            mask_prob = args.mask_prob,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            mlp_ratio = 4.0,
            embed_dim_decoder = args.embed_dim,
            num_layers_decoder = 2,
            num_heads_decoder = 6,
            norm_pix_loss = args.norm_pix_loss
        )
    elif args.ssl in ['simmim', 'SimMIM']:
        model = SimMIM(
            input_size = args.image_size,
            patch_size = args.patch_size,
            mask_prob = args.mask_prob,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            mlp_ratio = 4.0,
        )
    elif args.ssl in ['mbyol', 'MBYOL']:
        model = MBYOL(
            encoder = encoder,
            encoder_dim = args.embed_dim,
            feature_dim = 256,
            predict_dim = 256,
            n_mlplayers = 2,
            hidden_dim = 128,
            momentum = 0.996,
            image_size = args.image_size,
            patch_size = args.patch_size,
        )
    elif args.ssl in ['mdino', 'MDINO']:
        model = MDINO(
            encoder = encoder,
            encoder_dim = args.embed_dim,
            feature_dim = 256,
            n_mlplayers = 3,
            hidden_dim = 128,
            momentum = 0.996,
            image_size = args.image_size,
            patch_size = args.patch_size,
        )
    elif args.ssl in ['cgpt', 'CGPT']:
        model = CGPTARModel(
            num_classes = 0,
            input_size = args.image_size,
            patch_size = args.patch_size,
            in_chans = 3,
            embed_dim = args.embed_dim,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            mlp_ratio = 4.0,
        )
    else:
        raise NotImplementedError

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
    torch.backends.cudnn.benchmark = True

    args.knn_k = 200
    args.knn_t = 0.1

    args.writer = None
    if not args.distributed or args.rank == 0:
        with open(args.output_dir + "/args.json", 'w') as fid:
            default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
            json.dump(args.__dict__, fid, indent=2, default=default)
        args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))

    # start training
    print("=> begin training")
    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        utils.adjust_learning_rate(optimizer, epoch, args)
        lr = optimizer.param_groups[0]["lr"]

        # train for one epoch
        train_loss = train_epoch(train_loader, model, optimizer, epoch, args)
        # evaluate for one epoch
        real_model = model.module if args.ngpus > 1 else model
        test_accu1, test_accu5 = evaluate(memory_loader, test_loader, real_model, epoch, args)

        if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
            if not args.distributed or args.rank == 0:
                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.module.state_dict() if args.ngpus > 1 else model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, epoch + 1,
                    is_best=False,
                    save_dir=os.path.join(args.output_dir, f"checkpoint"))
        
        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar("Loss/train", train_loss, epoch)
            args.writer.add_scalar("Accu/test", test_accu1, epoch)
            args.writer.add_scalar("Misc/learning_rate", lr, epoch)


if __name__ == '__main__':

    args = parser.parse_args()

    args.data_dir = image_datasets[args.dataset]['data_dir']
    args.output_dir = image_datasets[args.dataset]['output_dir']

    output_prefix = f"{args.ssl}_vit"
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
