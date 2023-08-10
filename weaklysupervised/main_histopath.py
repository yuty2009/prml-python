
import os
import json
import tqdm
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import timm.optim.optim_factory as optim_factory

import common.torchutils as utils
from breakhisreader import BREAKHISPatchDataset
from bachreader import BACHPatchDataset
from attnmil import AttnMIL

datasets = {
    'breakhis': 'f:/medicalimages/breakhis',
    'bach': 'f:/medicalimages/bach/ICIAR2018_BACH_Challenge/Photos',
}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='breakhis', choices=datasets.keys())
parser.add_argument('--data_dir', type=str, 
                    default='')
parser.add_argument('--arch', type=str, default='attnmil')
parser.add_argument('--patch_size', type=int, default=28)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=50)
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
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
                    metavar='W', help='weight decay (default: 5e-2)',
                    dest='weight_decay')


def main():
    
    args = parser.parse_args()

    args.data_dir = datasets[args.dataset]
    args.output_dir = args.data_dir + '/output'

    output_prefix = f"{args.arch}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tf_train = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(90),
        transforms.ToTensor()
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor()
    ])

    if args.dataset == 'breakhis':
        tf_resize = transforms.Resize((448, 700)) # transforms.CenterCrop((448, 700))
        train_dataset = BREAKHISPatchDataset(
            args.data_dir, split='train', patch_size=args.patch_size, transform=tf_resize, transform_patch=tf_train,
        )
        test_dataset = BREAKHISPatchDataset(
            args.data_dir, split='test', patch_size=args.patch_size, transform=tf_resize, transform_patch=tf_test)
    elif args.dataset == 'bach':
        tf_resize = transforms.Resize((1488, 1984)) # transforms.CenterCrop((1488, 1984))
        train_dataset = BACHPatchDataset(
            args.data_dir, split='train', patch_size=args.patch_size, transform=tf_resize, transform_patch=tf_train)
        test_dataset = BACHPatchDataset(
            args.data_dir, split='test', patch_size=args.patch_size, transform=tf_resize, transform_patch=tf_test)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('Init Model')
    # CNN feature extractor
    feature_encoder = nn.Sequential(
        nn.Conv2d(3, 20, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(20, 50, kernel_size=5),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
    )
    feature_dim = 50 * 4 * 4 # for breakhis 28 * 28 patch_size
    # feature_dim = 50 * 28 * 28 # for bach 124 * 124 patch_size
    # feature_dim = 100 * 4 * 4 # for bach 124 * 124 patch_size
    # MIL model
    model = AttnMIL(feature_encoder, feature_dim=feature_dim, num_classes=2)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    args.writer = None
    with open(args.output_dir + "/args.json", 'w') as fid:
        default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
        json.dump(args.__dict__, fid, indent=2, default=default)
    args.writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "log"))
        
    for epoch in range(args.epochs):
        utils.adjust_learning_rate(optimizer, epoch, args)
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_accu = train_epoch(train_loader, model, criterion, optimizer, epoch, args)
        test_loss, test_accu = evaluate(test_loader, model, criterion, epoch, args)

        if args.output_dir and epoch > 0 and (epoch+1) % args.save_freq == 0:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, epoch + 1,
                is_best=False,
                save_dir=os.path.join(args.output_dir, f"checkpoint"))
            
        if hasattr(args, 'writer') and args.writer:
            args.writer.add_scalar("Loss/train", train_loss, epoch)
            args.writer.add_scalar("Loss/test", test_loss, epoch)
            args.writer.add_scalar("Accu/train", train_accu, epoch)
            args.writer.add_scalar("Accu/test", test_accu, epoch)
            args.writer.add_scalar("Misc/learning_rate", lr, epoch)

        print("Epoch: [{}/{}] train loss: {:.4f} train accu: {:.2f} test loss: {:.4f} test accu: {:.2f}".format(
            epoch+1, args.epochs, train_loss, train_accu, test_loss, test_accu
        ))


def train_epoch(data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    
    total_loss, total_num, total_correct = 0.0, 0, 0
    data_bar = tqdm.tqdm(data_loader)
    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        output = model(data)[0]
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_num += target.size(0)
        pred = output.argmax(dim=-1)

        total_correct += torch.sum((pred == target).float()).item()
        accu = 100 * total_correct / total_num

        info = "Train Epoch: [{}/{}] lr: {:.6f} Loss: {:.4f} ".format(
            epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss/len(data_loader))
        info += "Accu: {:.2f}".format(accu) 
        data_bar.set_description(info)
    
    return total_loss/len(data_loader), accu


def evaluate(data_loader, model, criterion, epoch, args):
    model.eval()

    ypreds, ytrues = [], []
    total_loss, total_num, total_correct = 0.0, 0, 0
    data_bar = tqdm.tqdm(data_loader)
    for data, target in data_bar:
        data = data.to(args.device)
        target = target.to(args.device)
        # compute output
        output = model(data)[0]
        loss = criterion(output, target)

        total_loss += loss.item()
        total_num += target.size(0)
        pred = output.argmax(dim=-1)
        ypreds.append(pred.cpu().numpy())
        ytrues.append(target.cpu().numpy())

        total_correct += torch.sum((pred == target).float()).item()
        accu = 100 * total_correct / total_num

        info = "Test  Epoch: [{}/{}] Loss: {:.4f} ".format(
            epoch, args.epochs, total_loss/len(data_loader))
        info += "Accu: {:.2f}".format(accu)
        data_bar.set_description(info)

    return total_loss/len(data_loader), accu

if __name__ == "__main__":

    main()
