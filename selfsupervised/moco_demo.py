
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import moco
import loader

class Arguments:
    def __init__(self):
        self.datapath = '/home/public/datasets/ImageNet'
        self.savepath = './checkpoint'
        self.gpu = None
        self.workers = 32
        self.arch = 'resnet50'
        self.moco_dim = 128
        self.moco_k = 65536
        self.moco_m = 0.999
        self.moco_t = 0.07
        self.mlp = False
        self.resume = None
        self.aug_plus = True
        self.cos = True
        self.lr = 0.03
        self.batch_size = 256
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.epochs = 200
        self.start_epoch = 0
        self.save_freq = 20
        self.verbose = False
        self.print_freq = 100


def main():

    args = Arguments()

    # create model
    print("Create model '{}'".format(args.arch))
    model = moco.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("Use all available GPUs")
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

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

    # Data loading code
    traindir = os.path.join(args.datapath, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    train_dataset = datasets.ImageFolder(
        traindir,
        loader.TwoCropsTransform(transforms.Compose(augmentation)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None, drop_last=True)

    # start training
    print("Start training")
    model.train()
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_accu1, train_accu5, train_loss = train_epoch(
            epoch, model, train_loader, criterion, optimizer, args)

        if epoch > 0 and epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, filename=os.path.join(args.savepath, 'checkpoint_{:04d}.pth.tar'.format(epoch)))

        print(f"Epoch: {epoch}, "
              f"Train loss: {train_loss:.3f}, accu@1: {train_accu1:.3f}, accu@5 {train_accu5:.3f}, "
              f"Epoch time = {time.time() - start_time:.1f} s")


def train_epoch(epoch, model, train_loader, criterion, optimizer, args):

    accus1, accus5, losses = [], [], []
    for batch_idx, (images, _) in enumerate(train_loader):

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

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

        if args.verbose and batch_idx % args.print_freq == 0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} Acc@1: {:.2f} Acc@5: {:.2f}"
                    .format(epoch, batch_idx, len(train_loader), loss.item(), accu1.item(), accu5.item()))

    return np.mean(accus1), np.mean(accus5), np.mean(losses)


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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()