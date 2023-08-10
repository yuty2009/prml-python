
import os
import tqdm
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

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
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=50)


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
        tf_crop = transforms.CenterCrop((448, 700))
        train_dataset = BREAKHISPatchDataset(
            args.data_dir, split='train', patch_size=args.patch_size, transform=tf_crop, transform_patch=tf_train,
        )
        test_dataset = BREAKHISPatchDataset(
            args.data_dir, split='test', patch_size=args.patch_size, transform=tf_crop, transform_patch=tf_test)
    elif args.dataset == 'bach':
        tf_crop = transforms.CenterCrop((1488, 1984))
        train_dataset = BACHPatchDataset(args.data_dir, split='train', patch_size=args.patch_size, transform=tf_train)
        test_dataset = BACHPatchDataset(args.data_dir, split='test', patch_size=args.patch_size, transform=tf_test)
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
    feature_dim = 50 * 4 * 4
    # MIL model
    model = AttnMIL(feature_encoder, feature_dim=feature_dim, num_classes=2)
    model = model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    for epoch in range(args.epochs):

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
