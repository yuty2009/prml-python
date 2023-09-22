
import tqdm
import torch
import common.distributed as dist


def train_epoch(data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    total_loss, total_num = 0.0, 0

    if not hasattr(args, 'weight_kl'): args.weight_kl = 1.0
    if not hasattr(args, 'topk'): args.topk = (1,)
    total_corrects = torch.zeros(len(args.topk), dtype=torch.float)

    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True
    data_bar = tqdm.tqdm(data_loader) if show_bar else data_loader

    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        output, loss_kl = model(data)
        loss_ce = criterion(output, target)
        loss = loss_ce * data.size(0) + args.weight_kl * loss_kl
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = dist.all_reduce(loss)
        total_loss += loss.item()
        total_num += data.size(0)
        preds = torch.argsort(output, dim=-1, descending=True)
        for i, k in enumerate(args.topk):
                total_corrects[i] += torch.sum((preds[:, 0:k] \
                    == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        accuks = 100 * total_corrects / total_num

        if show_bar:
            info = "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f} ".format(
                epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss/len(data_loader))
            info += ' '.join(["Acc@{}: {:.2f}".format(k, accuk) 
                            for k, accuk in zip(args.topk, accuks)])
            data_bar.set_description(info)
    
    return [total_loss/len(data_loader)] + [accuk for accuk in accuks]


def evaluate(data_loader, model, criterion, epoch, args):
    model.eval()
    total_loss, total_num = 0.0, 0

    if not hasattr(args, 'weight_kl'): args.weight_kl = 1.0
    if not hasattr(args, 'topk'): args.topk = (1,)
    total_corrects = torch.zeros(len(args.topk), dtype=torch.float)

    show_bar = False
    if not hasattr(args, 'distributed') or not args.distributed or \
       not hasattr(args, 'rank') or args.rank == 0:
        show_bar = True
    data_bar = tqdm.tqdm(data_loader) if show_bar else data_loader

    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        output, loss_kl = model(data)
        loss_ce = criterion(output, target)
        loss = loss_ce * data.size(0) + args.weight_kl * loss_kl

        loss = dist.all_reduce(loss)
        total_loss += loss.item()
        total_num += data.size(0)
        preds = torch.argsort(output, dim=-1, descending=True)
        for i, k in enumerate(args.topk):
                total_corrects[i] += torch.sum((preds[:, 0:k] \
                    == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        accuks = 100 * total_corrects / total_num

        if show_bar:
            info = "Test  Epoch: [{}/{}] Loss: {:.4f} ".format(
                epoch, args.epochs, total_loss/len(data_loader))
            info += ' '.join(["Acc@{}: {:.2f}".format(k, accuk) 
                            for k, accuk in zip(args.topk, accuks)])
            data_bar.set_description(info)
    
    return [total_loss/len(data_loader)] + [accuk for accuk in accuks]
