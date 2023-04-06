
import os
import copy
import json
import random
import datetime
import warnings
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import sys; sys.path.append(os.path.dirname(__file__)+"/../")
import common.torchutils as utils
from sslutils import *


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Self-Supervised Learning Benchmarks')
parser.add_argument('--ssl', default='deepcluster', type=str,
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
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')


def main(args):

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
        args.image_size = 32
        args.mean_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_dataset = datasets.CIFAR10(
            args.data_dir, train=True, download=True,
            transform=augment.get_transforms('train', args.image_size, args.mean_std)
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
        args.image_size = 96
        args.mean_std = ((0.4409, 0.4279, 0.3868), (0.2309, 0.2262, 0.2237))
        train_dataset = datasets.STL10(
            args.data_dir, split="train", download=True,
            transform=augment.get_transforms('train', args.image_size, args.mean_std)
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
        args.image_size = 224
        args.mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=augment.get_transforms('train', args.image_size, args.mean_std)
        )
        memory_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'train'),
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )
        test_dataset = datasets.ImageFolder(
            os.path.join(args.data_dir, 'val'),
            transform=augment.get_transforms('test', args.image_size, args.mean_std)
        )

    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=args.batch_size, shuffle=False,
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

    model = torch.nn.DataParallel(model)

    args.knn_k = 200
    args.knn_t = 0.1

    with open(args.output_dir + "/args.json", 'w') as fid:
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        json.dump(args.__dict__, fid, indent=2, default=default)
    args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))

    # start training
    print("=> begin training")
    for epoch in range(args.start_epoch, args.epochs):

        features = compute_features(memory_loader, model, args)
        assignments = kmeans_faiss(features, args.num_prototypes)

        reassigned_dataset = ReassignedDataset(train_dataset, assignments)

        train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            reassigned_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

        utils.adjust_learning_rate(optimizer, epoch, args)
        lr = optimizer.param_groups[0]["lr"]

        # train for one epoch
        train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, args)
        # evaluate for one epoch
        real_model = model.module if args.ngpus > 1 else model
        test_accu1, test_accu5 = evaluate_ssl(memory_loader, test_loader, real_model.encoder, epoch, args)

        if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
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


def train_epoch(data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    real_model = model.module if args.ngpus > 1 else model

    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True
    data_bar = tqdm.tqdm(data_loader) if show_bar else data_loader

    # create an optimizer for the prototypes layer
    optimizer_prototypes = torch.optim.SGD(
        real_model.prototypes.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_loss = 0.
    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)

        # compute output
        _, output = model(data)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        optimizer_prototypes.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_prototypes.step()

        # collect losses
        loss = dist.all_reduce(loss)
        total_loss += loss.item()

        # show progress
        if show_bar:
            info = "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f} ".format(
                epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss/len(data_loader))
            data_bar.set_description(info)
    
    return total_loss/len(data_loader)


def compute_features(data_loader, model, args):
    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True

    with torch.no_grad():
        data_bar = tqdm.tqdm(data_loader, desc='Computing features') if show_bar else data_loader
        feature_bank = []
        for data, _ in data_bar:
            feature, _ = model(data.to(args.device))
            feature = feature.detach()
            feature = torch.nn.functional.normalize(feature, dim=1)
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0) # [N, D]
    return feature_bank.cpu().numpy()


import faiss
def kmeans_faiss(x, k, n_iters=20, verbose=True):
    n_data, d = x.shape
    # faiss implementation of k-means
    clustering = faiss.Clustering(d, k)
    clustering.seed = np.random.randint(1234)
    clustering.verbose = verbose
    clustering.niter = n_iters
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)
    clustering.train(x, index)
    _, I = index.search(x, 1)
    indices = [int(n[0]) for n in I]
    return indices


def kmeans(x, k, n_iters=20, verbose=True):
    n_data = x.shape[0]
    assert n_data >= k, "number of samples should be larger than k"
    centroids = copy.deepcopy(x[:k])
    indices = torch.zeros(n_data, dtype=int, device=x.device)
    for it in range(n_iters):
        indices_old = copy.deepcopy(indices)
        dist = torch.mm(x, centroids.T)
        index_list = [[] for i in range(k)]
        progress_bar = tqdm.tqdm(range(n_data))
        progress_bar.set_description("Clustering [{}/{}]".format(it+1, n_iters))
        # update cluster assignments
        for i in progress_bar:
            ci = torch.argmin(dist[i])
            indices[i] = ci
            index_list[ci].append(i)
        # update centroids
        for j in range(k):
            if len(index_list[j]) > 0:
                centroids[j] = torch.mean(x[index_list[j],:], dim=0, keepdim=True)
        # stop iteration if assignments do not change
        if torch.equal(indices, indices_old):
            if verbose:
                print(f"clustering ended after {it+1} iterations")
            break
    if it >= n_iters and verbose:
        print(f"clustering ended due to maximum iterations")
    return indices


class ReassignedDataset(torch.utils.data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, dataset, pseudolabels, transform=None):
        assert len(pseudolabels) == len(dataset), "length does not match"
        self.transform = transform
        self.pseudolabels = pseudolabels
        self.images = []
        for idx in range(len(dataset)):
            path = dataset[idx][0]
            self.images.append(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        img = self.images[index]
        pseudolabel = self.pseudolabels[index]
        if isinstance(img, str):
            with open(img, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':

    args = parser.parse_args()

    output_prefix = f"ssl_{args.ssl}_{args.arch}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    args.ngpus = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)
