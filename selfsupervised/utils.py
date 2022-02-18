
import os
import math
import shutil
import numpy as np
import torch

def ssl_train_epoch(train_loader, model, criterion, optimizer, epoch, args):
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


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    accus1, accus5, losses = [], [], []
    for i, (images, target) in enumerate(train_loader):

        images = images.to(args.device)
        target = target.to(args.device)

        # compute output
        output = model(images)
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


def evaluate(eval_loader, model, criterion, args):
    model.eval()
    accus1, accus5, losses = [], [], []
    with torch.no_grad():
        for i, (images, target) in enumerate(eval_loader):

            images = images.to(args.device)
            target = target.to(args.device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            accu1, accu5 = accuracy(output, target, topk=(1, 5))
            losses.append(loss.item())
            accus1.append(accu1.item())
            accus5.append(accu5.item())

    return np.mean(accus1), np.mean(accus5), np.mean(losses)


def save_checkpoint(state, epoch, is_best, save_dir='./', prefix=''):
    checkpoint_path = os.path.join(
        save_dir, '{}_checkpoint_{:04d}.pth.tar'.format(prefix, epoch)
        )
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, '{}_model_best.pth.tar'.format(prefix))
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