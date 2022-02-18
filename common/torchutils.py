
import os
import torch


def train_epoch_ssl(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    if not hasattr(args, 'topk'): args.topk = (1,)
    accuks, losses = [[] for _ in args.topk], []
    for i, (images, _) in enumerate(train_loader):

        images[0] = images[0].to(args.device)
        images[1] = images[1].to(args.device)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)
        accuk = accuracy(output, target, topk=args.topk)
        losses.append(loss.item())
        [accuks[k].append(accu1.item()) for k, accu1 in enumerate(accuk)]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(args, 'verbose') and args.verbose and i % args.print_freq == 0:
            info = "Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} ".format(
                epoch, i, len(train_loader), loss.item()
                )
            info += ' '.join(["Acc@{}: {:.2f}".format(k, accu1) 
                            for k, accu1 in zip(args.topk, accuk)])
            print(info)
    
    res = [torch.tensor(accu1).mean() for accu1 in accuks] + [torch.tensor(losses).mean()]
    return res


def train_epoch(train_loader, model, criterion, optimizer, epoch, args):
    model.train()
    if not hasattr(args, 'topk'): args.topk = (1,)
    accuks, losses = [[] for _ in args.topk], []
    for i, (images, target) in enumerate(train_loader):

        images = images.to(args.device)
        target = target.to(args.device)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        accuk = accuracy(output, target, topk=args.topk)
        losses.append(loss.item())
        [accuks[k].append(accu1.item()) for k, accu1 in enumerate(accuk)]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if hasattr(args, 'verbose') and args.verbose and i % args.print_freq == 0:
            info = "Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f} ".format(
                epoch, i, len(train_loader), loss.item()
                )
            info += ' '.join(["Acc@{}: {:.2f}".format(k, accu1) 
                            for k, accu1 in zip(args.topk, accuk)])
            print(info)
    
    res = [torch.tensor(accu1).mean() for accu1 in accuks] + [torch.tensor(losses).mean()]
    return res


def evaluate(eval_loader, model, criterion, args):
    model.eval()
    if not hasattr(args, 'topk'): args.topk = (1,)
    accuks, losses = [[] for _ in args.topk], []
    with torch.no_grad():
        for i, (images, target) in enumerate(eval_loader):

            images = images.to(args.device)
            target = target.to(args.device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            accuk = accuracy(output, target, topk=args.topk)
            losses.append(loss.item())
            [accuks[k].append(accu1.item()) for k, accu1 in enumerate(accuk)]

    res = [torch.tensor(accu1).mean() for accu1 in accuks] + [torch.tensor(losses).mean()]
    return res


def save_checkpoint(state, epoch, is_best, save_dir='./', prefix='base'):
    checkpoint_path = os.path.join(
        save_dir, 'chkpt_{}_{:04d}.pth.tar'.format(prefix, epoch)
        )
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'best_{}.pth.tar'.format(prefix))
        torch.save(state, best_path)


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