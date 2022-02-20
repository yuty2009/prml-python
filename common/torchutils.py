
import os
import math
import torch
import torch.distributed as distributed

def train_epoch_ssl(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    if not hasattr(args, 'topk'): args.topk = (1,)
    loss_total = 0
    accuks = [[] for _ in args.topk]
    for i, (images, _) in enumerate(train_loader):

        images[0] = images[0].to(args.device)
        images[1] = images[1].to(args.device)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)
        accuk = accuracy(output, target, topk=args.topk)
        loss_total += loss.item()
        [accuks[k].append(accu1.item()) for k, accu1 in enumerate(accuk)]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if distributed.is_available() and distributed.is_initialized():
            loss = loss.data.clone()
            distributed.all_reduce(loss.div_(distributed.get_world_size()))

        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar("Loss/train_step", loss.item(), args.global_step)
            args.global_step += 1

        if hasattr(args, 'verbose') and args.verbose and i % args.print_freq == 0:
            info = "Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} ".format(
                epoch, i, len(train_loader), loss.item()
                )
            info += ' '.join(["Acc@{}: {:.2f}".format(k, accu1) 
                            for k, accu1 in zip(args.topk, accuk)])
            print(info)
    
    res = [torch.tensor(accu1).mean() for accu1 in accuks] + [loss_total/len(train_loader)]
    return res


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    if not hasattr(args, 'topk'): args.topk = (1,)
    loss_total = 0
    accuks = [[] for _ in args.topk]
    for i, (images, target) in enumerate(train_loader):

        images = images.to(args.device)
        target = target.to(args.device)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        accuk = accuracy(output, target, topk=args.topk)
        loss_total += loss.item()
        [accuks[k].append(accu1.item()) for k, accu1 in enumerate(accuk)]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if distributed.is_available() and distributed.is_initialized():
            loss = loss.data.clone()
            distributed.all_reduce(loss.div_(distributed.get_world_size()))

        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar("Loss/train_step", loss.item(), args.global_step)
            args.global_step += 1

        if hasattr(args, 'verbose') and args.verbose and i % args.print_freq == 0:
            info = "Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} ".format(
                epoch, i, len(train_loader), loss.item()
                )
            info += ' '.join(["Acc@{}: {:.2f}".format(k, accu1) 
                            for k, accu1 in zip(args.topk, accuk)])
            print(info)
    
    res = [torch.tensor(accu1).mean() for accu1 in accuks] + [loss_total/len(train_loader)]
    return res


def evaluate(eval_loader, model, criterion, args):
    model.eval()
    if not hasattr(args, 'topk'): args.topk = (1,)
    loss_total = 0
    accuks = [[] for _ in args.topk]
    with torch.no_grad():
        for i, (images, target) in enumerate(eval_loader):

            images = images.to(args.device)
            target = target.to(args.device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            accuk = accuracy(output, target, topk=args.topk)
            loss_total += loss.item()
            [accuks[k].append(accu1.item()) for k, accu1 in enumerate(accuk)]

    res = [torch.tensor(accu1).mean() for accu1 in accuks] + [loss_total/len(eval_loader)]
    return res


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.schedule in ['cos', 'cosine']:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    elif args.schedule in ['step', 'stepwise']:  # stepwise lr schedule
        for milestone in args.lr_drop:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, epoch, is_best, save_dir='./', prefix='base'):
    checkpoint_path = os.path.join(
        save_dir, 'chkpt_{}_{:04d}.pth.tar'.format(prefix, epoch)
        )
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_{}.pth.tar'.format(prefix))
        torch.save(state, best_path)


def convert_state_dict(state_dict):
    firstkey = next(iter(state_dict))
    if firstkey.startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.endswith('total_ops') and not k.endswith('total_params'):
                name = k[7:] # 7 = len('module.')
                new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


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

    pass